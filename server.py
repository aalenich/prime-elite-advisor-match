from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import json,random,threading,os,base64,hmac
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from train import features,SKILLS
ROOT=Path(__file__).resolve().parent
DEPLOY_MODE=os.getenv('DEPLOY_MODE','0')=='1'
APP_USERNAME=os.getenv('APP_USERNAME','demo')
APP_PASSWORD=os.getenv('APP_PASSWORD','')
if DEPLOY_MODE and (len(APP_PASSWORD)<12 or len(APP_USERNAME)<1):
    raise RuntimeError('Для публикации задайте APP_PASSWORD (не менее 12 символов) и APP_USERNAME в переменных окружения.')
MAN=pd.read_csv(ROOT/'data/managers.csv').fillna('')
CLI=pd.read_csv(ROOT/'data/clients.csv').fillna('')
ASSIGN=pd.read_csv(ROOT/'data/assignments.csv')
MODEL=CatBoostRegressor();MODEL.load_model(str(ROOT/'model/matcher.cbm'))
METRICS=json.loads((ROOT/'model/metrics.json').read_text())
LOCK=threading.Lock();STATE=ROOT/'data/live_assignments.json'
if not STATE.exists():STATE.write_text('[]')
REGIONS=sorted(MAN.region.unique().tolist());HOBBIES=['tennis','travel','reading','chess','art','music','skiing','running','cycling'];CHANNELS=['phone','chat','video']
LABELS={'bonds':'Облигации','ru_equities':'Акции РФ','global_markets':'Мировые рынки','deposits':'Вклады и накопления','lending':'Кредитование','margin':'Маржинальная торговля','structured_products':'Структурные продукты','tennis':'Теннис','travel':'Путешествия','reading':'Чтение','chess':'Шахматы','art':'Искусство','music':'Музыка','skiing':'Лыжи','running':'Бег','cycling':'Велоспорт','phone':'Телефон','chat':'Чат','video':'Видеосвязь'}
def state():return json.loads(STATE.read_text(encoding='utf-8'))
def loads():
    extra={}
    for a in state():extra[a['manager_id']]=extra.get(a['manager_id'],0)+1
    return {str(m.manager_id):int(m.initial_clients)+extra.get(str(m.manager_id),0) for m in MAN.itertuples()}
def managers():
    l=loads();out=[]
    for m in MAN.to_dict('records'):
        m['current_clients']=l[m['manager_id']];m['free_slots']=int(m['capacity'])-m['current_clients'];m['load_pct']=round(100*m['current_clients']/m['capacity'],1);out.append(m)
    return out
def new_client():
    segment=random.choices(['PRIME','ELITE'],[.66,.34])[0]
    return {'client_id':'NEW-'+str(random.randint(100000,999999)),'segment':segment,'region':random.choice(REGIONS),'capital_mln_rub':round(random.uniform(4,35) if segment=='PRIME' else random.uniform(35,250),1),'investment_experience':random.choice(['beginner','intermediate','advanced']),'needs':'|'.join(random.sample(SKILLS,random.randint(1,3))),'preferred_channel':random.choice(CHANNELS),'languages':'ru','hobbies':'|'.join(random.sample(HOBBIES,2)),'service_intensity':random.choice(['low','medium','high'])}
def recommend(c):
    eligible=[m for m in managers() if m['segment']==c['segment'] and m['free_slots']>0 and (set(str(c['languages']).split('_')) & set(str(m['languages']).split('_')))]
    if not eligible:return []
    rows=[]
    for m in eligible:
        d={'client_region':c['region'],'manager_region':m['region'],'client_channel':c['preferred_channel'],'manager_channel':m['preferred_channel'],'client_languages':c['languages'],'manager_languages':m['languages'],'client_hobbies':c['hobbies'],'manager_hobbies':m['hobbies'],'manager_initial_clients':m['current_clients'],'manager_capacity':m['capacity'],'manager_experience_years':m['experience_years'],'client_capital_mln_rub':c['capital_mln_rub'],'service_intensity':c['service_intensity'],'investment_experience':c['investment_experience'],'needs':c['needs']}
        for s in SKILLS:d['skill_'+s]=m['skill_'+s]
        rows.append(d)
    pred=MODEL.predict(features(pd.DataFrame(rows)))
    out=[]
    for m,p in zip(eligible,pred):
        common=list(set(c['hobbies'].split('|'))&set(m['hobbies'].split('|')))
        ns=c['needs'].split('|');strong=sorted(ns,key=lambda n:-m['skill_'+n])
        reasons=[f"Компетенция: {LABELS.get(strong[0],strong[0])} ({m['skill_'+strong[0]]}/10)"]
        if m['region']==c['region']:reasons.append('Совпадает регион')
        if m['preferred_channel']==c['preferred_channel']:reasons.append('Удобный канал связи')
        if common:reasons.append('Общие интересы: '+', '.join(LABELS.get(s,s) for s in common))
        reasons.append(f"Свободно мест: {m['free_slots']}")
        out.append({'manager':m,'score':round(float(np.clip(p,0,100)),1),'reasons':reasons})
    return sorted(out,key=lambda a:-a['score'])[:3]
def client_catalog(query):
    from urllib.parse import parse_qs
    params=parse_qs(query)
    segment=params.get('segment',[''])[0]
    region=params.get('region',[''])[0]
    search=params.get('search',[''])[0].strip().lower()
    page=max(1,min(int(params.get('page',['1'])[0]),10000))
    size=25
    # Original clients plus clients created through the matching workflow.
    base=CLI.copy()
    if segment:base=base[base.segment==segment]
    if region:base=base[base.region==region]
    if search:base=base[base.client_id.str.lower().str.contains(search,regex=False)]
    base=base.copy()
    base['manager_id']=base.client_id.map(dict(zip(ASSIGN.client_id,ASSIGN.manager_id))).fillna('')
    base['assignment_status']=base.manager_id.apply(lambda v:'Назначен' if v else 'Не назначен')
    extras=[]
    for a in state():
        c=a.get('client')
        if not c:continue
        if segment and c.get('segment')!=segment:continue
        if region and c.get('region')!=region:continue
        if search and search not in c.get('client_id','').lower():continue
        extras.append({**c,'manager_id':a['manager_id'],'assignment_status':'Назначен (демо)'})
    total=len(base)+len(extras)
    start=(page-1)*size
    if start<len(base):
        rows=base.iloc[start:start+size].to_dict('records')
        rows+=extras[:max(0,size-len(rows))]
    else:rows=extras[start-len(base):start-len(base)+size]
    return {'items':rows,'total':total,'page':page,'page_size':size,'pages':max(1,(total+size-1)//size)}
def stats():
    mm=managers();ss=state();return {'total_managers':len(mm),'total_clients':len(CLI)+len(ss),'new_assignments':len(ss),'occupied':sum(m['current_clients'] for m in mm),'capacity':sum(m['capacity'] for m in mm),'segments':[{'name':s,'managers':sum(m['segment']==s for m in mm),'clients':sum(m['current_clients'] for m in mm if m['segment']==s),'capacity':sum(m['capacity'] for m in mm if m['segment']==s)} for s in ['PRIME','ELITE']],'regions':[{'name':r,'clients':sum(m['current_clients'] for m in mm if m['region']==r),'capacity':sum(m['capacity'] for m in mm if m['region']==r)} for r in REGIONS],'metrics':METRICS,'recent':ss[-10:][::-1]}
class Handler(SimpleHTTPRequestHandler):
    def authenticated(self):
        if not APP_PASSWORD:
            return True
        auth=self.headers.get('Authorization','')
        if not auth.startswith('Basic '):
            return False
        try:
            raw=base64.b64decode(auth[6:],validate=True).decode('utf-8')
            username,password=raw.split(':',1)
            return hmac.compare_digest(username,APP_USERNAME) and hmac.compare_digest(password,APP_PASSWORD)
        except (ValueError,UnicodeDecodeError):
            return False
    def require_auth(self):
        if self.authenticated():
            return False
        self.send_response(401)
        self.send_header('WWW-Authenticate','Basic realm="Prime & Elite Demo", charset="UTF-8"')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length','0')
        self.end_headers()
        return True
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def __init__(self,*a,**k):super().__init__(*a,directory=str(ROOT/'static'),**k)
    def sendjson(self,obj,status=200):
        b=json.dumps(obj,ensure_ascii=False,default=str).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.require_auth():return
        p=urlparse(self.path).path
        if p=='/api/managers':return self.sendjson(managers())
        if p=='/api/clients':return self.sendjson(client_catalog(urlparse(self.path).query))
        if p=='/api/stats':return self.sendjson(stats())
        if p=='/api/client':return self.sendjson(new_client())
        if p=='/api/meta':return self.sendjson({'regions':REGIONS,'skills':SKILLS,'labels':LABELS})
        return super().do_GET()
    def do_POST(self):
        if self.require_auth():return
        try:
            data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            p=urlparse(self.path).path
            if p=='/api/recommend':
                c=data['client'];return self.sendjson({'client':c,'recommendations':recommend(c)})
            if p=='/api/assign':
                c=data['client'];mid=data['manager_id']
                with LOCK:
                    if any(a['client_id']==c['client_id'] for a in state()):return self.sendjson({'error':'Этот клиент уже назначен'},409)
                    allowed=[r['manager']['manager_id'] for r in recommend(c)]
                    if mid not in allowed:return self.sendjson({'error':'Менеджер недоступен или не входит в актуальный ТОП-3'},409)
                    a=state();a.append({'client_id':c['client_id'],'manager_id':mid,'segment':c['segment'],'region':c['region'],'client':c});STATE.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
                return self.sendjson({'ok':True,'manager_id':mid,'stats':stats()})
            if p=='/api/reset':
                with LOCK:STATE.write_text('[]',encoding='utf-8')
                return self.sendjson({'ok':True})
            return self.sendjson({'error':'Неизвестный метод'},404)
        except Exception as e:return self.sendjson({'error':str(e)},400)
if __name__=='__main__':
    port=int(os.getenv('PORT','8000'))
    host='0.0.0.0' if DEPLOY_MODE else '127.0.0.1'
    print(f'Prime & Elite server: http://{host}:{port}')
    ThreadingHTTPServer((host,port),Handler).serve_forever()
