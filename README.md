# camera_contamination_simulation
Simulation Framework for Camera Contamination Simulation

In this framework, we introduce innovative simulation methods to artificially generate three distinct types of camera lens soiling, designed to aid in the testing and development of next-generation camera-based ADAS systems. By leveraging a lightweight CNN model and utilizing the WoodScape dataset in [1], we simulated realistic contamination patterns. Opaque soiling was crafted using mask fusion methods, while transparent soiling was modeled using Perlin Noise, producing visually convincing artifacts on clean images. For generating water droplets, we developed a classifier to distinguish between two visual scenarios within the droplet. For larger droplets, we use a heuristic method to simulate various lens blockages. We evaluate this simulation framework using a real stereo dataset from \cite{stereodata} with clear and soiled images.

To assess the impact of these synthetic camera contaminations, we trained several U-Net-based models using different proportions of augmented data. Our evaluation, conducted on both the WoodScape test set dirtycam [2] and an additional dataset collected in Japan—representing entirely different environmental conditions—demonstrates that models trained with augmented data consistently outperformed the baseline. This indicates that the artificially generated soiling closely mimics real-world conditions in terms of shape and behavior. The results from both tables confirm that the augmentation method is highly effective, leading to improvements in performance metrics across all tested models, validating the robustness of our approach in enhancing model generalization.

The whole theoretical background of the framework is documented in the following publications:

WACV 2025 - "Clarity amidst Blur: A Deterministic Method for Synthetic Generation of Water Droplets on Camera Lenses"

SSCi 2025 - "Beyond the Smudge: Simulating Opaque and Transparent Automotive Camera Lens Soiling"





## Examples
See usage_example jupyter Notebook


## Data
Uitilized data is available in [1] and [2]


## References

[1] M. Uˇriˇc´aˇr, G. Sistu, H. Rashed, A. Vobeck´y, V. R. Kumar, P. Kˇr´ıˇzek, F. B¨urger, and S. Yogamani, “Let’s get dirty: Gan based data augmentation for camera lens soiling detection in autonomous driving,” in 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 766–775.
Link: https://woodscape.valeo.com/woodscape/

[2] Y. Chang, “Soiling dataset,” https://github.com/Leozyc-waseda/SoilingDataset, 2020.