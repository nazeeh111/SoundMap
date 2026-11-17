"""Actual dashboard callbacks with controllable synthetic measurement generations."""
import sys, time, threading, unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
from bokeh.document import Document
from bokeh.models import Button, Div, ColumnDataSource
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acoustic-camera'))
from ui import dashboard as ui

class FakeProcessor:
 def __init__(self):
  self.model_revision=4; self.beam_revision=7
  self.data={'x':[7.0],'y':[8.0],'z':[9.0],'s':[4.0]}
  self.beam={'results':None,'max_x':[3.0],'max_y':[4.0],'max_s':5.0}
  self.release=threading.Event(); self.entered=threading.Event();self.mode=None
 def get_model_snapshot(self):return self.model_revision,self.data.copy()
 def get_beamforming_snapshot(self):return self.beam_revision,self.beam.copy()
 def start_model(self):self.mode='model';self.model_stop_event=threading.Event();self.entered.set()
 def start_beamforming(self):self.mode='beamforming';self.beamforming_stop_event=threading.Event();self.entered.set()
 def stop_model(self):self.release.wait(2)
 def stop_beamforming(self):self.release.wait(2)

def until(fn):
 t=time.monotonic()+2
 while not fn():
  if time.monotonic()>t:raise AssertionError('timed out')
  time.sleep(.003)

class DashboardGeneration(unittest.TestCase):
 def setUp(self):
  self.doc=Document();self.patch=patch.object(ui,'curdoc',return_value=self.doc);self.patch.start()
  self.p=FakeProcessor();self.d=ui.Dashboard.__new__(ui.Dashboard);d=self.d
  d.document=self.doc;d.processor=self.p;d.measurement=ui.MeasurementOwner(self.p,gate=threading.Lock())
  d.deviation_cds=ColumnDataSource(dict(time=[9.],x_deviation=[1.],y_deviation=[2.],z_deviation=[3.]))
  d.measurement_button=Button(label='Start');d.measurement_status=Div();d.method=0
  d.config={'ui.start_text':'Start','ui.stop_text':'Stop'}
  d.estimation_callback=d.overflow_callback=d.beamforming_callback=d.lifecycle_callback=None
  d.estimation_update_interval=d.overflow_update_interval=d.beamforming_update_interval=150
  d._enable_widgets=d._disable_widgets=lambda:None
  self.model_draw=[];self.beam_draw=[]
  d.acoustic_camera_plot=S(update_plot_model=lambda data:self.model_draw.append(data),update_plot_beamforming=lambda data:self.beam_draw.append(data),update_plot_beamforming_dots=lambda data:self.beam_draw.append(data),second_view=S(visible=True),model_cds=ColumnDataSource(dict(x=[],y=[],z=[],s=[])),beamforming_cds=ColumnDataSource(dict(beamformer_data=[])),beamforming_dot_cds=ColumnDataSource(dict(x=[],y=[])))
  d.model_params_column=S(visible=False);d.overflow_status=Div();d.deviation_plot=S(visible=True);d.method_selector=S(active=0)
  d.coordinates_display=Div();d.level_display=Div();d.real_x=d.real_y=d.real_z=0
 def tearDown(self):
  self.p.release.set();self.d.measurement.stop()
  if self.d.measurement._thread:self.d.measurement._thread.join(2)
  self.patch.stop()
 def start(self):
  self.d.start_measurement();until(lambda:self.d.measurement.snapshot()[0]=='running')
 def stop(self):
  self.d.stop_measurement();self.p.release.set();self.d.measurement._thread.join(2);self.d._poll_measurement()
 def test_model_freshness_stop_and_queued_mode(self):
  d=self.d;self.start();d.update_estimations()
  self.assertEqual(list(d.deviation_cds.data['time']),[]);self.assertEqual(self.model_draw,[])
  self.p.model_revision+=1;d.update_estimations();self.assertEqual(len(self.model_draw),1);self.assertEqual(len(d.deviation_cds.data['time']),1)
  d.stop_measurement();self.assertEqual(len(d.deviation_cds.data['time']),1)
  d.method_selector.active=1;d.toggle_method('active',0,1)
  self.assertEqual(d.method_selector.active,0);self.assertEqual(d.method,0);self.assertIsNone(d.beamforming_callback)
  self.p.model_revision+=1;d.update_estimations();self.assertEqual(len(self.model_draw),1)
  self.p.release.set();d.measurement._thread.join(2);d._poll_measurement()
  self.assertEqual(list(d.deviation_cds.data['time']),[]);self.assertIsNone(d.estimation_callback);self.assertIsNone(d.beamforming_callback)
  self.p.release.clear();self.start();d.update_estimations();self.assertEqual(len(self.model_draw),1);self.assertEqual(list(d.deviation_cds.data['time']),[])
  self.p.model_revision+=1;d.update_estimations();self.assertEqual(len(self.model_draw),2)
 def test_beamforming_freshness_and_mode_switch_in_idle(self):
  d=self.d;d.method_selector.active=1;d.toggle_method('active',0,1);self.assertIsNone(d.beamforming_callback)
  self.start();d.update_beamforming();self.assertEqual(self.beam_draw,[])
  self.p.beam_revision+=1;d.update_beamforming();self.assertEqual(len(self.beam_draw),1)
  d.stop_measurement();self.p.beam_revision+=1;d.update_beamforming();self.assertEqual(len(self.beam_draw),1)
  self.p.release.set();d.measurement._thread.join(2);d._poll_measurement();self.assertIsNone(d.beamforming_callback)
 def test_process_gate_prevents_second_session_until_real_cleanup(self):
  other=FakeProcessor();other_owner=ui.MeasurementOwner(other,gate=self.d.measurement._gate)
  self.start();self.assertFalse(other_owner.start('model'))
  self.d.stop_measurement();self.assertFalse(other_owner.start('model'))
  self.p.release.set();self.d.measurement._thread.join(2)
  self.assertTrue(other_owner.start('model'));until(lambda:other_owner.snapshot()[0]=='running')
  other_owner.stop();other.release.set();other_owner._thread.join(2);self.assertFalse(other_owner._thread.is_alive())
 def test_teardown_while_starting_defers_other_session(self):
  p=self.p;started=threading.Event();allow=threading.Event();original=p.start_model
  def slow_start():
   started.set();allow.wait(2);original()
  p.start_model=slow_start
  owner=ui.MeasurementOwner(p,gate=threading.Lock());other=FakeProcessor();second=ui.MeasurementOwner(other,gate=owner._gate)
  self.assertTrue(owner.start('model'));self.assertTrue(started.wait(1))
  self.assertTrue(owner.stop());self.assertFalse(second.start('model'))
  allow.set();until(lambda:p.entered.is_set());until(lambda:p.model_stop_event.is_set())
  self.assertFalse(second.start('model'));p.release.set();owner._thread.join(2)
  self.assertTrue(second.start('model'));until(lambda:second.snapshot()[0]=='running')
  second.stop();other.release.set();second._thread.join(2)
 def test_real_dashboard_registers_session_destroy_stop(self):
  p=self.p;p.dev=None;p.mics=S(mpos=[[0],[0]])
  config={'layout.video.width':320,'layout.video.height':240,'app_settings.estimation_update_interval':100,'app_settings.beamforming_update_interval':100,'app_settings.camera_update_interval':100,'app_settings.stream_update_interval':100,'app_default_settings.x':0,'app_default_settings.y':0,'app_default_settings.z':0}
  with patch.object(ui,'AcousticCameraPlot',return_value=S(fig=Div())):
   d=ui.Dashboard(config,p,False,(1,1))
  callbacks=list(self.doc.session_destroyed_callbacks);self.assertEqual(len(callbacks),1)
  self.assertTrue(d.measurement.start('model'));until(lambda:d.measurement.snapshot()[0]=='running')
  other=FakeProcessor();other_owner=ui.MeasurementOwner(other)
  callbacks[0](None)
  self.assertEqual(d.measurement.snapshot()[0],'stopping');self.assertFalse(other_owner.start('model'))
  p.release.set();d.measurement._thread.join(2);self.assertTrue(other_owner.start('model'))
  until(lambda:other_owner.snapshot()[0]=='running');other_owner.stop();other.release.set();other_owner._thread.join(2)

if __name__=='__main__':unittest.main()
