# SEMI
Segmented Error Minimisation (SEMI) for Robust training of deep learning models with non-linear shifts in reference data.
Harry J. Davies∗ , Yuyang Miao∗, Amir Nassibi∗, Morteza Khaleghimeybodi†, Danilo P. Mandic∗
∗Imperial College London, UK
†Meta Reality Labs Research, USA

This work is accepted for presentation at ICASSP 2024 in Seoul, Korea.

Abstract: Time series regression models are typically trained using the mean squared error (MSE) and thus rely critically on time-aligned reference data. However, the MSE loss is often inadequate when processing real-world data, such as physiological signals, as misalignment between two signals can cause a large change in MSE, thus severely inhibiting convergence. This is regularly compounded by time varying drifts between different modalities and such as photoplethysmography, electrocardiography, blood pressure and respiration.
Indeed, these exhibit fluctuating time delays even when taken from the same individual across different body positions and at different times of the day. To this end, we introduce the concept of segmented error minimisation (SEMI), a new loss function which accounts for differing time delays among the variables. The SEMI is examined both through simulations, and via a denoising convolutional autoencoder with synthetic data. It is then finally verified in the real-world application of the denoising of wearable photoplethysmography with a reference signal.



