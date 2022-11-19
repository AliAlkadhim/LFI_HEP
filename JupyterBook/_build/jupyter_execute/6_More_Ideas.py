#!/usr/bin/env python
# coding: utf-8

# ------------------
# -----------------
# # More Discussions
# 
# One simple example which is paradoxical is the following: suppose we have our usual likelihood 
# $$L(\theta, \nu)= \frac{e^{-(\theta+\nu)} (\theta+\nu)^N }{N !} \ \frac{e^{-\nu} \nu^M}{M !} \tag{20} $$ and we observe $N=M=0$ so that the likelihood becomes $ L(\theta, \nu)= e^{-(\theta+\nu)} e^{-\nu} $. Suppose that now we wish to make an inference on $\theta$. It should make sense that if we know for a fact that there is no background events (since $M=0$), that the likelihood should not depend on the mean background (since there is no backround to begin with). [Feldman and Cousins](https://arxiv.org/pdf/physics/9711021v2.pdf) were aware of this issue and assigned the problem to be the incorrect interpretation of the intervals as Bayesian intervals.
# 
# A very closely related problem is the existence of a signal and background densities, say both depending on POI $\theta$, and the likelihood will be given by
# 
# $$ L(\theta) = \prod_{i=1}^{N_{obs}} \theta S(x_i) + (1-\theta) B(x_i)$$
# where the likelihood $L(\theta)$ is the probability for obtaining the observation $x$ from either a signal or background distribution, where $\theta$ is an unknown proportion of signal (since real events come as a mixture of signal and background), and we would like to infer about the value of this parameter of interest. In the presense of nuissance parameter $\nu$ the likelihood becomes
# 
# $$ L(\theta,\nu) = \prod_{i=1}^{N_{obs}} \theta S(x_i|\nu) + (1-\theta) B(x_i|\nu)$$
# 
# 
# # More ideas
# 
# 1. More real world physics scenrios
#     * We can test this technique in a real case physics LFI scenario by generating pythia data on the fly for signal and background with unknown nuissance parameter and making inferences on the POI
# 
# 2. Likelihood regions and confidence intervals from confidence sets of two or more parameters. Confidence sets of potentially any dimension could be mapped to (potentially) any lower dimension
# 
# 3. Using more test statistics, motivated by [Asymptotic formulae for likelihood-based tests of new physics](https://arxiv.org/pdf/1007.1727.pdf)
#    - This is especially useful if we want to use it for physics discovery of a positive or null signal, since the Cranmer paper deal with pragmatics of these new test statistics for physics discoveries.

# In[ ]:




