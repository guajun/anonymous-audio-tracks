"""Optional real-time TensorBoard events with actual measurement timestamps."""
from pathlib import Path
import json
import numpy as np


class Telemetry:
    def __init__(self,root,run,metadata):
        self.writer=None
        if root is not None:
            from torch.utils.tensorboard import SummaryWriter
            directory=Path(root)/run
            if directory.exists() and any(directory.iterdir()):
                raise ValueError("TensorBoard run directory must be new")
            self.writer=SummaryWriter(str(directory),flush_secs=10)
            self.writer.add_text('metadata/config',json.dumps(metadata,ensure_ascii=False,indent=2),0)

    def scalars(self,prefix,values,step):
        if self.writer:
            for key,value in values.items():
                if value is not None and isinstance(value,(int,float)) and not isinstance(value,bool):
                    self.writer.add_scalar(f'{prefix}/{key}',value,step)

    def evaluation(self,prefix,result,step):
        self.scalars(prefix,result,step)
        songs=result['songs']
        for key in ('extra_active_fraction','mean_extra_active_count','source_count_mae','association_entropy','association_null_mass','association_capacity_residual','cycle_eligible_fraction'):
            values=[s[key] for s in songs if key in s]
            if values:self.scalars(prefix,{key:float(np.mean(values))},step)
        for key in ('raw_A_mean','raw_A_max','tracked_A_mean','tracked_A_max','raw_A_sum_mean','tracked_A_sum_mean',
                    'raw_A_active_mean','tracked_A_active_mean','raw_A_silent_mean','tracked_A_silent_mean','transported_mass_ratio','pit_optimal_count'):
            values=[s[key] for s in songs if s.get(key) is not None]
            if values:self.scalars(prefix,{key:float(np.mean(values))},step)
        for key in ('pregate_A_mean','pregate_A_max','gate_zero_fraction','gate_all_zero_frames','fragment_count','fragment_columns',
                    'direct_anchor_frames','expired_endpoints','local_link_pairs','reference_short_gap_pairs','short_gap_both_detected',
                    'short_gap_pit_stitched','short_gap_pit_stitch_fraction','diagnostic_short_gap_fragment_breaks','reference_long_gap_pairs'):
            values=[s[key] for s in songs if s.get(key) is not None]
            if values:self.scalars(prefix,{key:float(np.mean(values))},step)
        for key in ('onset_mae_seconds','offset_mae_seconds','frame_f1'):
            values=[s['events'][key] for s in songs if s['events'][key] is not None]
            if values:self.scalars(prefix+'/events',{key:float(np.mean(values))},step)
        for key in ('matched_events','missed_events','extra_events'):
            self.scalars(prefix+'/events',{key:sum(s['events'][key] for s in songs)},step)
        self.scalars(prefix,{'pit_ambiguous_fraction':float(np.mean([s['assignment_ambiguous'] for s in songs]))},step)
        if self.writer:self.writer.flush()

    def close(self):
        if self.writer:self.writer.close()

    def curves(self,directory,step):
        if not self.writer:return
        files=sorted(Path(directory).glob('*val*.npz'))
        if not files:return
        with np.load(files[0]) as curves:
            layout={}
            for source in range(curves['target'].shape[1]):
                prefix=f'envelope/val/{files[0].stem}/step-{step}/source-{source+1}'
                tags=[prefix+'/reference',prefix+'/tracked']
                for time,truth,predicted in zip(curves['center_times'],curves['target'][:,source],curves['predicted'][:,source]):
                    self.writer.add_scalar(tags[0],float(truth),round(float(time)*1000))
                    self.writer.add_scalar(tags[1],float(predicted),round(float(time)*1000))
                layout[f'source-{source+1}']=['Multiline',tags]
            if 'raw_candidates' in curves:
                prefix=f'envelope/val/{files[0].stem}/step-{step}/all-candidates'
                tags=[prefix+'/raw-mean',prefix+'/tracked-mean']
                for time,raw,tracked in zip(curves['center_times'],curves['raw_candidates'].mean(1),curves['tracked_candidates'].mean(1)):
                    self.writer.add_scalar(tags[0],float(raw),round(float(time)*1000))
                    self.writer.add_scalar(tags[1],float(tracked),round(float(time)*1000))
                layout['all-candidates']=['Multiline',tags]
            self.writer.add_custom_scalars({'Original-track milliseconds':layout})
        self.writer.flush()
