f1_ml_pre_manifest = 'keniack/f1-ml-pre'
f2_ml_train_manifest = 'keniack/f2-ml-train'
f3_ml_eval_manifest = 'keniack/f3-ml-eval'
pi_function = 'f1-ml-pre'
f2_ml_function = 'f2-ml-train'
f3_ml_function = 'f3-ml-eval'



all_ai_images = [
    (f1_ml_pre_manifest, '200M', 'x86'),
    (f1_ml_pre_manifest, '2000M', 'amd64'),
    (f1_ml_pre_manifest, '300M', 'arm32v7'),
    (f1_ml_pre_manifest, '300M', 'arm32'),
    (f1_ml_pre_manifest, '300M', 'arm'),
    (f1_ml_pre_manifest, '540M', 'aarch64'),
    (f1_ml_pre_manifest, '540M', 'arm64'),

    (f2_ml_train_manifest, '350M', 'x86'),
    (f2_ml_train_manifest, '430M', 'amd64'),
    (f2_ml_train_manifest, '320M', 'arm32v7'),
    (f2_ml_train_manifest, '320M', 'arm32'),
    (f2_ml_train_manifest, '320M', 'arm'),
    (f2_ml_train_manifest, '440M', 'aarch64'),
    (f2_ml_train_manifest, '540M', 'arm64'),

    (f3_ml_eval_manifest, '350M', 'x86'),
    (f3_ml_eval_manifest, '430M', 'amd64'),
    (f3_ml_eval_manifest, '320M', 'arm32v7'),
    (f3_ml_eval_manifest, '320M', 'arm32'),
    (f3_ml_eval_manifest, '320M', 'arm'),
    (f3_ml_eval_manifest, '440M', 'aarch64'),
    (f3_ml_eval_manifest, '540M', 'arm64'),

]
