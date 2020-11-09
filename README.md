# VFWS : VoiceFilter Without Speaker



Using Voicefilter to noise surppression only  without targeted speaker embedding.

<img src=https://github.com/kooBH/VFWS/blob/master/resources/VFWS.PNG>

# Conclusion
Not Good

#  Result

log magnitude, L1 loss, oneCycleLR, step 40k, epoch 3
<img src=https://github.com/kooBH/VFWS/blob/master/resources/VFWS_tensorboard.PNG>
<img src=https://github.com/kooBH/VFWS/blob/master/resources/VFWS_v2_step40k.gif>

Mask for speech is not solid, smudging speech TF.

Based on https://arxiv.org/abs/1810.04826  
and  https://github.com/mindslab-ai/voicefilter


