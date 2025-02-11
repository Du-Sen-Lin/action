_base_ = 'timesformer_spaceOnly_8xb8-8x32x1-15e_ucf101-rgb.py'

model = dict(backbone=dict(attention_type='divided_space_time'))
