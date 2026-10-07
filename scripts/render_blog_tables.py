#!/usr/bin/env python3
"""Render blog tables as PNG + SVG previews. Requires Pillow and Arial fonts.
Run from any directory with python3 scripts/render_blog_tables.py.
Set TABLE_FONT_DIR to a directory containing Arial.ttf and Arial Bold.ttf.
"""
from pathlib import Path
import os,re,html
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/fine_tuning.md'
OUT=ROOT/'docs/images/tables'
FONT_DIR=Path(os.environ.get('TABLE_FONT_DIR','/System/Library/Fonts/Supplemental'))
FONTS={}
def font(size,bold=False):
 key=(size,bold)
 if key not in FONTS:FONTS[key]=ImageFont.truetype(str(FONT_DIR/('Arial Bold.ttf' if bold else 'Arial.ttf')),size)
 return FONTS[key]
def plain(s):
 s=re.sub(r'\[([^\]]+)\]\([^)]+\)',lambda m:m[1]+' ',s)
 s=re.sub(r'<br\s*/?>',' ',s)
 s=s.replace('**','').replace('`','').replace('\\','').replace('▶', 'Play')
 return re.sub(r'\s+',' ',s).strip()
def wrap(s,width,size,bold=False):
 f=font(size,bold);lines=[];line=''
 for word in s.split():
  if line and f.getlength(line+' '+word)>width:lines.append(line);line=''
  while f.getlength(word)>width:
   if '-' in word:
    part,rest=word.split('-',1)
    if f.getlength(part+'-')<=width:
     if line:lines.append(line);line=''
     lines.append(part+'-');word=rest;continue
   if line:lines.append(line);line=''
   n=1
   while n<len(word) and f.getlength(word[:n+1])<=width:n+=1
   lines.append(word[:n]);word=word[n:]
  if word:line=(line+' '+word).strip()
 if line:lines.append(line)
 return lines or ['']
SPECS=[
 ('Editorial overview','Publication brief and review notes',[.20,.80]),
 ('Review and publication','Deadlines, reviewers, and awareness',[.32,.68]),
 ('Two skills, one adaptation workflow','What you can do with each skill',[.50,.50]),
 ('Choose the ASR architecture','How each decoder works and where it fits',[.15,.42,.43]),
 ('Know your training ingredients','Source pools, label quality, and purpose',[.15,.29,.34,.22]),
 ('Keep Gold data roles separate','Human-verified ATCO2 splits',[.23,.13,.14,.50]),
 ('Listen to ATC speech','Audio examples and reference transcripts · links in the Markdown table',[.24,.29,.47]),
 ('Four questions guide the experiments','Objectives and the models evaluated',[.27,.48,.25]),
 ('Gold labels and English replay','Comparison 1 · WER (%) · lower is better',[.15,.35,.125,.125,.125,.125]),
 ('More training on Jacktol alone','Comparison 2 · Nemotron · WER (%) · lower is better',[.16,.30,.135,.135,.135,.135]),
 ('Broad adaptation, then focused refinement','Comparison 3 · WER (%) · lower is better',[.15,.30,.11,.11,.11,.11,.11]),
 ('Add aviation language at decoding time','Comparison 4 · fixed acoustic checkpoints · WER (%) · lower is better',[.20,.24,.08,.12,.12,.12,.12]),
]
def render(index,raw):
 title,subtitle,weights=SPECS[index-1]
 rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in raw]
 rows.pop(1)
 assert all(len(r)==len(weights) for r in rows)
 W=1800;M=48;PAD=20;SIZE=24;LINE=34
 # Result tables: size columns from their contents so every data cell is one line.
 if index>=9:
  widths=[]
  for ci in range(len(weights)):
   body=max(font(SIZE,ci==0 or re.fullmatch(r'\*\*[^*]+\*\*',r[ci]) is not None).getlength(plain(r[ci])) for r in rows[1:])
   header=max(font(22,True).getlength(word) for word in plain(rows[0][ci]).split())
   widths.append(int(max(body,header)+2*PAD+2))
  W=max(W,sum(widths)+2*M)
  extra=W-2*M-sum(widths)
  targets=list(range(2,len(widths)))
  for n in range(extra):widths[targets[n%len(targets)]]+=1
 else:
  widths=[round((W-2*M)*w) for w in weights];widths[-1]=W-2*M-sum(widths[:-1])
  if index!=3:
   label_width=int(max(font(SIZE,True).getlength(plain(r[0])) for r in rows[1:])+2*PAD+2)
   if widths[0]<label_width<=560:
    need=label_width-widths[0];rest=sum(widths[1:])
    widths=[label_width]+[w-round(need*w/rest) for w in widths[1:]]
    widths[-1]=W-2*M-sum(widths[:-1])
 INNER=W-2*M
 layout=[]
 for ri,row in enumerate(rows):
  cells=[]
  for ci,cell in enumerate(row):
   bold=ri==0 or (ci==0 and index!=3) or re.fullmatch(r'\*\*[^*]+\*\*',cell) is not None
   size=22 if ri==0 else SIZE
   # A small, bounded reduction can save a short entry from an awkward wrap.
   if ri>0 and index<9 and len(plain(cell))<85:
    for candidate in range(size,21,-1):
     if font(candidate,bold).getlength(plain(cell))<=widths[ci]-2*PAD:
      size=candidate;break
   lines=wrap(plain(cell),widths[ci]-2*PAD,size,bold)
   if index>=9 and ri>0:assert len(lines)==1,(index,ri,ci,lines)
   cells.append((lines,size,bold))
  height=max(len(c[0]) for c in cells)*LINE+2*PAD
  layout.append((cells,height))
 H=170+sum(h for _,h in layout)+76
 im=Image.new('RGB',(W,H),'#F3F5F1');d=ImageDraw.Draw(im)
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title"><title id="title">{html.escape(title)}</title><g font-family="Arial, Helvetica, sans-serif">']
 def rect(x,y,w,h,fill):
  d.rectangle((x,y,x+w,y+h),fill=fill)
  svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>')
 def txt(x,y,s,size,color,bold=False,right=False,center=False):
  f=font(size,bold)
  if center:x-=f.getlength(s)/2
  elif right:x-=f.getlength(s)
  d.text((x,y),s,font=f,fill=color,anchor='ls')
  svg.append(f'<text x="{x:.2f}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
 rect(0,0,W,H,'#F3F5F1');rect(M,35,64,6,'#76B900')
 txt(M,96,title,43,'#182019',True);txt(M,137,subtitle,24,'#596555')
 numeric_columns={ci for ci in range(len(weights)) if all(re.fullmatch(r'[\d.,% —]+(?: h)?',plain(row[ci])) for row in rows[1:])}
 y=170
 for ri,(cells,h) in enumerate(layout):
  baseline=ri>0 and (plain(rows[ri][0]).lower().startswith('pretrained baseline') or any('fresh greedy control' in cell.lower() for cell in rows[ri]))
  fill='#182019' if ri==0 else '#EAF3DD' if baseline else '#FFFFFF' if ri%2 else '#F7F9F5'
  rect(M,y,INNER,h,fill)
  if baseline:rect(M,y,5,h,'#76B900')
  x=M
  for ci,(lines,size,bold) in enumerate(cells):
   color='#FFFFFF' if ri==0 else '#397000' if '**' in rows[ri][ci] and re.fullmatch(r'\*\*[\d.]+%\*\*',rows[ri][ci]) else '#202A22'
   numeric=ci in numeric_columns
   for li,line in enumerate(lines):txt(x+widths[ci]/2 if numeric else x+PAD,y+(h-len(lines)*LINE)/2+25+li*LINE,line,size,color,bold,center=numeric)
   x+=widths[ci]
  y+=h
  rect(M,y-1,INNER,1,'#CBD3C6')
 txt(M,H-28,'ASR FINE-TUNING STUDY',18,'#596555',True)
 txt(W-M,H-28,f'TABLE {index:02}',18,'#596555',True,True)
 svg.append('</g></svg>')
 stem=f'table-{index:02}'
 im.save(OUT/(stem+'.png'),optimize=True)
 (OUT/(stem+'.svg')).write_text('\n'.join(svg)+'\n')
 return f'<!-- styled-table:{index:02} -->\n[![Styled table: {title}](images/tables/{stem}.png)](images/tables/{stem}.svg)\n<!-- /styled-table -->'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 source=DOC.read_text()
 source=re.sub(r'\n*<!-- styled-table:\d+ -->.*?<!-- /styled-table -->\n*','\n\n',source,flags=re.S)
 lines=source.splitlines();result=[];i=0;count=0
 while i<len(lines):
  if lines[i].startswith('|'):
   block=[]
   while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
   count+=1;result.extend(block+['',render(count,block)])
  else:result.append(lines[i]);i+=1
 assert count==len(SPECS),(count,len(SPECS))
 DOC.write_text('\n'.join(result)+'\n')
 print(f'Rendered {count} tables as PNG and SVG; original Markdown tables retained.')
if __name__=='__main__':main()
