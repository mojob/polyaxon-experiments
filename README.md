# Polyaxon-experiments

Code for blog post at:

https://blog.mojob.io/other/sharing-learning-experimenting-with-polyaxon/

# To run training
```
pip install -r requirements.txt
```

```
 POLYAXON_NO_OP=TRUE python -u train.py --epochs=32 \
  --log_learning_rate=-3 \
  --optimizer="adam" \
  --first_layer_output=32 \
  --kernel_size=3 \
  --pool_size=2 \
  --loss_metric="categorical_crossentropy" \
  --epochs=2 \
  --layers="conv2d:64,maxpooling2d,dropout:0.2,flatten,dense:128,dropout:0.5"
```
(POLYAXON_NO_OP turns of experiment logging when running training locally)
