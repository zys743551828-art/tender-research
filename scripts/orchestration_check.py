"""LangGraph boundary experiment with fictional nodes, no model or network calls.

Uses actual LangGraph with PostgresSaver on a real PostgreSQL 16 (separate runtime database from the business DB). Persistent checkpoints are operational state,
not company records. Each child is a separate Python process.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

import psycopg
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]
os.environ['LANGSMITH_TRACING']='false'
os.environ['LANGCHAIN_TRACING_V2']='false'
import socket

# This experiment deliberately rejects all network connections, including tracing.
_connect=socket.socket.connect
def deny_network(self,address,*args,**kwargs):
    host=address[0] if isinstance(address,tuple) else address   # only loopback and unix sockets (the local PostgreSQL) are allowed
    if self.family==socket.AF_UNIX or host in ('127.0.0.1','::1','localhost'):return _connect(self,address,*args,**kwargs)
    raise RuntimeError('network_disabled_for_fixture')
socket.socket.connect=deny_network
from importlib.metadata import version

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field


class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    approved:bool
    reason:str=Field(min_length=1)
class Candidate(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    claim:str
    evidence_ids:list[str]=Field(min_length=1)
    nature:Literal['response_conflict','needs_facts']
class State(TypedDict,total=False):
    snapshot_id:str
    input_sha256:str
    graph_version:str
    stage:str
    evidence_ids:list[str]
    finding:dict
    approved:bool
    revision:str

def graph(connection,saver,fault=False):
    # 计数与故障注入使用独立连接，避免与检查点存储的流水线连接互相占用。
    side=psycopg.connect(connection.info.dsn,autocommit=True,row_factory=dict_row)
    connection=side
    connection.execute('CREATE TABLE IF NOT EXISTS fixture_counts(thread_id TEXT,node TEXT,n INTEGER,PRIMARY KEY(thread_id,node))')
    connection.execute('CREATE TABLE IF NOT EXISTS fixture_faults(thread_id TEXT PRIMARY KEY)')
    def count(node,config):
        thread=config['configurable']['thread_id']
        connection.execute('INSERT INTO fixture_counts VALUES (%s,%s,1) ON CONFLICT(thread_id,node) DO UPDATE SET n=fixture_counts.n+1',(thread,node))
        return thread
    def rules(state,config):count('rules',config);return {'stage':'rules'}
    def retrieve(state,config):count('retrieve',config);return {'stage':'retrieve','evidence_ids':['fiction_tender_page_12','fiction_body_page_17']}
    def diagnose(state,config):
        thread=count('diagnose',config)
        if fault and not connection.execute('SELECT 1 FROM fixture_faults WHERE thread_id=%s',(thread,)).fetchone():
            connection.execute('INSERT INTO fixture_faults VALUES (%s)',(thread,));raise RuntimeError('injected_node_failure')
        result=Candidate(claim='Fictional overall schedule differs between two sources',evidence_ids=state['evidence_ids'],nature='response_conflict')
        return {'finding':result.model_dump(),'stage':'diagnosed'}
    def review(state,config):
        count('review_entered',config)
        answer=interrupt({'kind':'engineering_review','finding':state['finding']},response_schema=Decision)
        return {'approved':answer.approved,'stage':'reviewed'}
    def revise(state,config):count('revise',config);return {'revision':'Fictional local revision based on confirmed evidence','stage':'revised'}
    def recheck(state,config):count('recheck',config);return {'stage':'await_final_engineering_check'}
    builder=StateGraph(State)
    for name,node in [('rules',rules),('retrieve',retrieve),('diagnose',diagnose),('review',review),('revise',revise),('recheck',recheck)]:builder.add_node(name,node)
    builder.add_edge(START,'rules');builder.add_edge('rules','retrieve');builder.add_edge('retrieve','diagnose');builder.add_edge('diagnose','review')
    builder.add_conditional_edges('review',lambda s:'revise' if s['approved'] else END,{'revise':'revise',END:END})
    builder.add_edge('revise','recheck');builder.add_edge('recheck',END)
    return builder.compile(checkpointer=saver)

FROZEN={'snapshot_id':'snapshot_fiction','input_sha256':'a'*64,'graph_version':'fixture-graph-v1'}
def child(db,thread,phase,body):
    with psycopg.connect(db,autocommit=True,row_factory=dict_row) as conn:
        saver=PostgresSaver(conn);saver.setup();app=graph(conn,saver,fault=phase in ['fault','recover'])
        config={'configurable':{'thread_id':thread},'recursion_limit':16}
        if phase=='start' or phase=='fault':result=app.invoke(FROZEN,config,durability='sync')
        elif phase=='recover':result=app.invoke(None,config,durability='sync')
        else:
            saved=app.get_state(config)
            if not saved.values:raise RuntimeError('unknown_graph_run')
            expected=body.pop('expected_input',FROZEN)
            if any(saved.values.get(k)!=v for k,v in expected.items()):raise RuntimeError('frozen_input_mismatch')
            if not saved.interrupts:raise RuntimeError('no_pending_interrupt')
            result=app.invoke(Command(resume=body),config,durability='sync')
        pending=app.get_state(config).interrupts
        counts={r['node']:r['n'] for r in conn.execute('SELECT node,n FROM fixture_counts WHERE thread_id=%s',(thread,))}
        return {'pending_review':bool(pending),'state':{k:v for k,v in result.items() if k!='__interrupt__'},'counts':counts}

if len(sys.argv)>1 and sys.argv[1]=='--child':
    try:print(json.dumps(child(sys.argv[2],sys.argv[3],sys.argv[4],json.loads(sys.argv[5])),ensure_ascii=False))
    except Exception as e:print(json.dumps({'error_type':type(e).__name__,'error':str(e)},ensure_ascii=False));sys.exit(1)
    sys.exit(0)
checks=[]
def check(name,fn):
    try:fn();checks.append({'name':name,'status':'passed'})
    except Exception as e:checks.append({'name':name,'status':'failed','error':str(e)})
def require(condition):
    if not condition:raise AssertionError('fixture assertion failed')
sys.path.insert(0,str(ROOT/'reference'))
BASE_DSN=os.environ.get('PG_DSN','host=localhost port=5432 user=postgres')
def admin():
    return psycopg.connect(BASE_DSN+' dbname=postgres',autocommit=True)
def drop_db(name):
    with admin() as a:a.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')
import uuid

dbname='verify_lg_'+uuid.uuid4().hex[:10]
with admin() as a:a.execute(f'CREATE DATABASE {dbname}')
db=BASE_DSN+f' dbname={dbname}'
if True:
    def run(thread,phase,body=None,expect_error=None):
        p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',db,thread,phase,json.dumps(body or {})],capture_output=True,text=True,check=False)
        result=json.loads(p.stdout.strip())
        if expect_error:require(p.returncode!=0 and expect_error in result['error'])
        else:require(p.returncode==0)
        return result
    started=run('run_a','start')
    check('真实LangGraph在人工复核前暂停且尚未修订',lambda:require(started['pending_review'] and 'revision' not in started['state']))
    check('候选包含定位证据且不包含分数或官方结论',lambda:require(set(started['state']['finding'])=={'claim','evidence_ids','nature'} and len(started['state']['finding']['evidence_ids'])==2))
    check('缺少复核理由不能恢复',lambda:run('run_a','resume',{'approved':True},'reason'))
    check('复核请求不能注入其他字段',lambda:run('run_a','resume',{'approved':True,'reason':'Fixture review','official_score':'5'},'official_score'))
    check('版本变化不能沿原快照恢复',lambda:run('run_a','resume',{'approved':True,'reason':'Fixture review','expected_input':{**FROZEN,'snapshot_id':'snapshot_other'}},'frozen_input_mismatch'))
    completed=run('run_a','resume',{'approved':True,'reason':'Confirmed fictional evidence'})
    check('新进程从文件断点恢复并进入末次工程核查',lambda:require(not completed['pending_review'] and completed['state']['stage']=='await_final_engineering_check'))
    check('恢复不重复已完成的规则检索和诊断节点',lambda:require(all(completed['counts'].get(n)==1 for n in ['rules','retrieve','diagnose','revise','recheck'])))
    check('人工中断节点从头执行且前段没有业务写入',lambda:require(completed['counts']['review_entered']>=2))
    run('run_b','start');rejected=run('run_b','resume',{'approved':False,'reason':'Insufficient fictional evidence'})
    check('工程驳回进入终态且不形成修订',lambda:require(not rejected['pending_review'] and 'revision' not in rejected['state'] and 'revise' not in rejected['counts']))
    check('不同run_id不会恢复他人断点',lambda:run('run_missing','resume',{'approved':True,'reason':'Fixture'},'unknown_graph_run'))
    run('run_fault','fault',expect_error='injected_node_failure');recovered=run('run_fault','recover')
    check('节点故障后新进程恢复，不重跑已完成前置节点',lambda:require(recovered['pending_review'] and recovered['counts']['rules']==1 and recovered['counts']['retrieve']==1 and recovered['counts']['diagnose']==2))
    check('复核完成的流程不能再次用同一中断恢复',lambda:run('run_a','resume',{'approved':True,'reason':'Again'},'no_pending_interrupt'))
drop_db(dbname)
report={'record_id':'orchestration-check','recorded_at':datetime.now(UTC).isoformat(),'scope':'actual_langgraph_with_postgres_saver_and_fictional_nodes_separate_processes','versions':{name:version(name) for name in ['langgraph','langgraph-checkpoint-postgres','psycopg','langchain-core','pydantic']},'checks':checks,'passed':sum(c['status']=='passed' for c in checks),'failed':sum(c['status']=='failed' for c in checks),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['没有调用DeepSeek或处理企业资料，不能证明工程诊断与评分质量','PostgresSaver 在同机单实例 PostgreSQL 16 上验证独立进程恢复；未验证跨主机、并发租约、主备切换或与 Kafka 联动恢复','版本与中断守卫属于试验适配层，不是框架自带公司权限；业务审核记录与正式发布仍需目标环境联调','外部请求响应丢失仍可能重复调用和计费；本测试不证明供应商恰好调用一次','模型、检索与修订节点为虚构返回，不证明本项目端到端算法可用']}
(ROOT/'results/orchestration-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'failed':report['failed'],'failures':[c for c in checks if c['status']=='failed']},ensure_ascii=False))
if report['failed']:sys.exit(1)
