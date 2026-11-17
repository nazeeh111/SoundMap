"""Read exact offline outputs and render geometry/band maps. No Acoular call."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
P=Path(__file__).resolve().parent;V=json.loads((P/'provenance.json').read_text())
for name,h in V['source_sha256'].items():assert hashlib.sha256((P.parents[1]/name).read_bytes()).hexdigest()==h
G=np.asarray(V['microphones_m']);invalid=np.asarray(V['excluded_zero_based_channels']);valid=np.ones(len(G),bool);valid[invalid]=False
maps={}
for band in V['bands']:
 a=np.load(P/band['npz']);power=a['power'];xyz=a['coordinates_m'];assert power.shape==(13,13) and xyz.shape==(3,13,13);assert np.allclose(xyz[2],-.68)
 maps[band['nominal_hz']]=(power,xyz,json.loads((P/band['receipt']).read_text()))
ink='#1d2934';muted='#526274';blue='#284d90';paper='#f4f6f9';red='#ae3c32'
def render(phone=False):
 w,h=(440,1720) if phone else (1100,800)
 with plt.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'svg.fonttype':'none','svg.hashsalt':'soundmap-recorded-bands'}):
  fig=plt.figure(figsize=(w/100,h/100),dpi=100,facecolor=paper);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
  def txt(x,y,s,size=18,c=ink,weight='normal',ha='left'):return ax.text(x,y,s,fontsize=size*.72,color=c,weight=weight,ha=ha,va='center')
  m=24 if phone else 40;txt(m,32,'SOUNDMAP / RECORDED-DATA BEAMFORMING',12,muted,'bold')
  if phone:txt(m,79,'One recording.',32,weight='bold');txt(m,119,'Two frequency bands.',32,weight='bold');txt(m,159,'54 valid channels · 0.3125 s analyzed',17)
  else:txt(m,86,'One recording. Two frequency bands.',44,weight='bold');txt(m,134,'56 recorded channels → 54 analyzed · 0.3125 s · conventional Acoular beamforming',20)
  def axes_at(x,y,size):return fig.add_axes([x/w,(h-y-size)/h,size/w,size/h])
  gx,gy,gs=(88,246,240) if phone else (62,248,260)
  txt(m if phone else 40,gy-56,'ORDERED MICROPHONE GEOMETRY',13,blue,'bold');txt(m if phone else 40,gy-29,'XML coordinates · array plane z = 0 m',15)
  g=axes_at(gx,gy,gs);g.scatter(G[valid,0],G[valid,1],s=22,c=blue,label='54 valid channels',gid='microphones-valid');g.scatter(G[invalid,0],G[invalid,1],s=55,marker='x',c=red,linewidth=1.8,label='excluded 1,7',gid='microphones-excluded');g.set(xlim=(-.75,.75),ylim=(-.75,.75),aspect='equal',xticks=[-.6,0,.6],yticks=[-.6,0,.6]);g.set_xlabel('XML x / m',fontsize=10);g.set_ylabel('XML y / m',fontsize=10);g.tick_params(labelsize=9);g.grid(color='#e1e6ed',lw=.6);g.spines[['top','right']].set_visible(False)
  txt(m if phone else 40,gy+gs+54,'Excluded channel indices: 1, 7 (zero-based)',14,muted)
  txt(m if phone else 40,gy+gs+80,'No coordinate mirror or camera overlay',14,muted)
  for idx,hz in enumerate([4000,5000]):
   xx,yy,sz=((88,660+idx*410,264) if phone else (420+idx*350,248,260));tx=m if phone else xx-20
   power,xyz,report=maps[hz];band=V['bands'][idx]
   txt(tx,yy-56,f'Nominal {hz/1000:g} kHz / third octave',17,blue,'bold');txt(tx,yy-29,'Bins: '+', '.join(f'{f:g}' for f in band['actual_bin_centres_hz'])+' Hz',14,muted)
   a=axes_at(xx,yy,sz);relative=10*np.log10(np.maximum(power/power.max(),1e-12));im=a.pcolormesh(xyz[0,:,0],xyz[1,0,:],relative.T,shading='nearest',cmap='magma',norm=Normalize(-20,0,clip=True),rasterized=False);im.set_gid(f'map-{hz}-cells')
   peak=report['peaks'][0]['position_m'];a.scatter([peak[0]],[peak[1]],marker='+',s=130,c='white',linewidth=2,gid=f'peak-{hz}');a.set(aspect='equal',xticks=[-.6,-.3,0],yticks=[-.3,0,.3]);a.set_xlabel('XML x / m',fontsize=10);a.set_ylabel('XML y / m',fontsize=10);a.tick_params(labelsize=9)
   txt(tx,yy+sz+53,f'Grid maximum: ({peak[0]:g}, {peak[1]:g}) m',15,weight='bold')
  cy=1470 if phone else 625;cx=84 if phone else 436;cw=275 if phone else 550
  cax=fig.add_axes([cx/w,(h-cy-14)/h,cw/w,14/h]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-20,-15,-10,-5,0]);cb.ax.tick_params(labelsize=10);cb.set_label('Relative level / dB re each map maximum',fontsize=11,color=ink)
  if phone:
   txt(24,1590,'Supplied plane z = −0.68 m · grid 0.05 m',15)
   txt(24,1620,'FFT spacing 400 Hz · each map has its own 0 dB',14,muted)
   txt(24,1650,'No independent SPL or localization validation',14,muted)
   txt(24,1685,'Ennes Sarradj (2021) · CC BY 4.0',14,muted)
  else:
   txt(40,632,'Same array frame, supplied scan plane:',16);txt(40,662,'z = −0.68 m (not estimated)',18,weight='bold')
   txt(40,716,'13 × 13 grid points per map · grid spacing 0.05 m · FFT spacing 400 Hz · sound speed 346.04 m/s',17)
   txt(40,750,'Each map is normalized separately; colors do not compare absolute band levels. Display range: −20 to 0 dB.',16)
   txt(40,780,'Ennes Sarradj (2021), CC BY 4.0 · Wind-tunnel data · No independent localization accuracy, absolute SPL or flow correction.',14,muted)
  name='recorded-bands-mobile' if phone else 'recorded-bands';fig.savefig(P/(name+'.png'),dpi=100,facecolor=paper);fig.savefig(P/(name+'.svg'),facecolor=paper,metadata={'Date':None});plt.close(fig)
  svg=(P/(name+'.svg')).read_text();svg=re.sub(r'font-family:[^;\"]+',"font-family: 'DejaVu Sans', Arial, sans-serif",svg);svg=svg.replace('<svg ','<svg role="img" aria-labelledby="sm-title sm-desc" ',1);i=svg.index('>',svg.index('<svg '))+1;svg=svg[:i]+'<title id="sm-title">One recording. Two frequency bands.</title><desc id="sm-desc">Actual public wind-tunnel recording and ordered 56-microphone geometry. Excluding channels 1 and 7 leaves 54. Conventional beamforming maps nominal 4 kHz and 5 kHz third-octave bands on a supplied z=-0.68 m plane. Maxima are (-0.25,-0.1) m and (-0.25,0.05) m. Each map has its own peak-relative dB reference. No independently measured localization accuracy or absolute SPL.</desc>'+svg[i:];(P/(name+'.svg')).write_text(svg)
render();render(True)
