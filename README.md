# ROSE Lab Dual Norm

## Training
```bash
python train.py config/dual_norm.yaml --DEVICE cuda:0
```

## Testing
```bash
python test.py config/dual_norm.yaml
```

## Testing Offical Weight
save to official weight as net_160.pth in the `Dual_Norm_Official` folder in `checkpoint`
```bash
python test.py config/dual_norm_official.yaml 18530
```