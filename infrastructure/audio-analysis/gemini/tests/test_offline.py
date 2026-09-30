"""Exercise upload/call/save/cleanup with a fake client; no external requests."""
import json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze

class OfflineTest(unittest.TestCase):
 def test_request_and_cleanup(self):
  self.check_request(False)
 def test_auxiliary_request_and_cleanup(self):
  self.check_request(True)
 def test_custom_input_labels(self):
  self.check_request(False, 60)
 def check_request(self, auxiliary, duration=30):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);audio=root/'a.wav';audio.write_bytes(b'fixture');prompt=root/'p.txt';prompt.write_text('listen' if duration==30 else 'full_clip lasts 60 seconds; reference is original 1-2s.',encoding='utf-8')
   uploads=[];deleted=[];calls=[]
   def upload(file):
    name=f'files/{len(uploads)}';uploads.append(file)
    return SimpleNamespace(name=name,uri='test://'+name,mime_type='audio/wav',state=SimpleNamespace(name='ACTIVE'))
   def create(**kwargs):
    calls.append(kwargs);return SimpleNamespace(output_text=json.dumps({'schema':'anonymous-audio-tracks/v1','duration_s':duration,'audio_access':True,'target_identified':False,'sources':[]}))
   client=SimpleNamespace(files=SimpleNamespace(upload=upload,delete=lambda name:deleted.append(name)),interactions=SimpleNamespace(create=create))
   class Context:
    def __enter__(self):return client
    def __exit__(self,*args):pass
   args=['analyze.py','--audio',str(audio),'--reference',str(audio),'--prompt',str(prompt)]
   if duration!=30:args.extend(['--duration',str(duration)])
   if auxiliary:args.extend(['--auxiliary',str(audio)])
   with patch.object(analyze,'ROOT',root),patch.object(sys,'argv',args),patch.dict(os.environ,{'GEMINI_API_KEY':'test-not-a-real-key'}),patch.object(analyze.genai,'Client',return_value=Context()):
    self.assertEqual(analyze.main(),0)
   count=3 if auxiliary else 2
   self.assertEqual(len(uploads),count);self.assertEqual(len(deleted),count)
   self.assertEqual(calls[0]['model'],'gemini-3.8-flash')
   texts=[x['text'] for x in calls[0]['input'] if x['type']=='text']
   self.assertEqual(texts[1:3],['reference','full_clip'])
   self.assertFalse(any('10.5-12' in text or '0-30s' in text for text in texts))
   if duration==60:self.assertIn('reference is original 1-2s.',texts[0])
   self.assertEqual(sum(x['type']=='audio' for x in calls[0]['input']),count)
   self.assertEqual(len(list(root.glob('runs/*/tracks.json'))),1)
 def test_bad_time(self):
  with self.assertRaises(AssertionError):analyze.validate({'schema':'anonymous-audio-tracks/v1','duration_s':30,'audio_access':True,'sources':[{'id':'S1','onsets':[{'time_s':31,'confidence':1,'timing_uncertainty_ms':10}],'active_intervals':[],'anchors':[]}]})
if __name__=='__main__':unittest.main()
