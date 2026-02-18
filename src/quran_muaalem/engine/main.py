"""
Create preprocessing deployment that extracts each features form every input

Create Model Deplymnet its only responlbility to run model infrence with dynmanic batching and return logits wiche runs on the GPU

Creat PostProcessing Deplyment that outputs only the text for `phonemes` level and the ids for the rest of level as a dict
without using aligment modules of the muaalem

Every request comes to the preprocessing then bathces accumlates and the model run inference then post processing per evvery output
"""
