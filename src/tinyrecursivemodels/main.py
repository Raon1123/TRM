def build_arc1():
    import sys
    sys.argv = ['build_arc_dataset.py', '--input-file-prefix', 'kaggle/combined/arc-agi', '--output-dir', 'data/arc1concept-aug-1000', '--subsets', 'training', 'evaluation', 'concept', '--test-set-name', 'evaluation']
    from dataset.build_arc_dataset import cli
    cli()

def build_arc2():
    import sys
    sys.argv = ['build_arc_dataset.py', '--input-file-prefix', 'kaggle/combined/arc-agi', '--output-dir', 'data/arc2concept-aug-1000', '--subsets', 'training2', 'evaluation2', 'concept', '--test-set-name', 'evaluation2']
    from dataset.build_arc_dataset import cli
    cli()

def build_sudoku():
    import sys
    sys.argv = ['build_sudoku_dataset.py', '--output-dir', 'data/sudoku-extreme-1k-aug-1000', '--subsample-size', '1000', '--num-aug', '1000']
    from dataset.build_sudoku_dataset import cli
    cli()

def build_maze():
    import sys
    sys.argv = ['build_maze_dataset.py']
    from dataset.build_maze_dataset import cli
    cli()

def pretrain_arc1():
    import sys
    sys.argv = ['pretrain.py', 'arch=trm', 'data_paths=[data/arc1concept-aug-1000]', 'arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=4', '+run_name=pretrain_att_arc1', 'ema=True']
    from pretrain import launch
    launch()

def pretrain_arc2():
    import sys
    sys.argv = ['pretrain.py', 'arch=trm', 'data_paths=[data/arc2concept-aug-1000]', 'arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=4', '+run_name=pretrain_att_arc2', 'ema=True']
    from pretrain import launch
    launch()

def pretrain_sudoku_mlp_t():
    import sys
    sys.argv = ['pretrain.py', 'arch=trm', 'data_paths=[data/sudoku-extreme-1k-aug-1000]', 'evaluators=[]', 'epochs=50000', 'eval_interval=5000', 'lr=1e-4', 'puzzle_emb_lr=1e-4', 'weight_decay=1.0', 'puzzle_emb_weight_decay=1.0', 'arch.mlp_t=True', 'arch.pos_encodings=none', 'arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=6', '+run_name=pretrain_mlp_t_sudoku', 'ema=True']
    from pretrain import launch
    launch()

def pretrain_sudoku_att():
    import sys
    sys.argv = ['pretrain.py', 'arch=trm', 'data_paths=[data/sudoku-extreme-1k-aug-1000]', 'evaluators=[]', 'epochs=50000', 'eval_interval=5000', 'lr=1e-4', 'puzzle_emb_lr=1e-4', 'weight_decay=1.0', 'puzzle_emb_weight_decay=1.0', 'arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=6', '+run_name=pretrain_att_sudoku', 'ema=True']
    from pretrain import launch
    launch()

def pretrain_maze():
    import sys
    sys.argv = ['pretrain.py', 'arch=trm', 'data_paths=[data/maze-30x30-hard-1k]', 'evaluators=[]', 'epochs=50000', 'eval_interval=5000', 'lr=1e-4', 'puzzle_emb_lr=1e-4', 'weight_decay=1.0', 'puzzle_emb_weight_decay=1.0', 'arch.L_layers=2', 'arch.H_cycles=3', 'arch.L_cycles=4', '+run_name=pretrain_att_maze30x30', 'ema=True']
    from pretrain import launch
    launch()