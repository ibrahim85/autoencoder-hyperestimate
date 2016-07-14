nut --exec='cd /src; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_00_pretrainingConvCifar10_solver.sh > 16-06-15_15h49m18s_0_00_logs.txt 2>&1; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_10_pretrainClassificationFrozen_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_00_pretrainingConvCifar10_iter_2000.caffemodel > 16-06-15_15h49m18s_0_10_logs.txt 2>&1; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_11_pretrainClassification_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_10_pretrainClassificationFrozen_iter_3000.caffemodel > 16-06-15_15h49m18s_0_11_logs.txt 2>&1; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_20_reconstructIncremental1_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_11_pretrainClassification_iter_20000.caffemodel > 16-06-15_15h49m18s_0_20_logs.txt 2>&1; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_30_reconstructIncremental2_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_20_reconstructIncremental1_iter_2000.caffemodel > 16-06-15_15h49m18s_0_30_logs.txt 2>&1; time python2 VisualizeReconstructionOfLayer.py snapshots/16-06-15_15h49m18s_0_40_reconstructFull_iter_30000.caffemodel prototxt/16-06-15_15h49m18s_0_40_reconstructFull_net.sh; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_40_reconstructFull_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_30_reconstructIncremental2_iter_2000.caffemodel > 16-06-15_15h49m18s_0_40_logs.txt 2>&1; time caffe test --weights=snapshots/16-06-15_15h49m18s_0_40_reconstructFull_iter_30000.caffemodel --model=prototxt/16-06-15_15h49m18s_0_40_reconstructFull_net.sh --gpu=0 > 16-06-15_15h49m18s_0_40_logs_tests.txt 2>&1; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_50_reconstructFullFC0_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_40_reconstructFull_iter_30000.caffemodel > 16-06-15_15h49m18s_0_50_logs.txt 2>&1; time caffe test --weights=snapshots/16-06-15_15h49m18s_0_50_reconstructFullFC0_iter_30000.caffemodel --model=prototxt/16-06-15_15h49m18s_0_50_reconstructFullFC0_net.sh --gpu=0 > 16-06-15_15h49m18s_0_50_logs_tests.txt 2>&1; time python2 VisualizeReconstructionOfLayer.py snapshots/16-06-15_15h49m18s_0_50_reconstructFullFC0_iter_30000.caffemodel prototxt/16-06-15_15h49m18s_0_50_reconstructFullFC0_net.sh; python2 VisualizeReconstructionOfLayer.py snapshots/16-06-15_15h49m18s_0_60_reconstructFullFC0unfrozen_iter_30000.caffemodel prototxt/16-06-15_15h49m18s_0_60_reconstructFullFC0unfrozen_net.sh; time caffe train --solver=prototxt/16-06-15_15h49m18s_0_60_reconstructFullFC0unfrozen_solver.sh --weights=snapshots/16-06-15_15h49m18s_0_50_reconstructFullFC0_iter_30000.caffemodel > 16-06-15_15h49m18s_0_60_logs.txt 2>&1; time caffe test --weights=snapshots/16-06-15_15h49m18s_0_60_reconstructFullFC0unfrozen_iter_30000.caffemodel --model=prototxt/16-06-15_15h49m18s_0_60_reconstructFullFC0unfrozen_net.sh --gpu=0 > 16-06-15_15h49m18s_0_60_logs_tests.txt 2>&1'