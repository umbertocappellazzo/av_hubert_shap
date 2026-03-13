## Shapley Value computations for AV-HuBERT

This repository contains the code to compute the audio/video SHAP contributions for the AV-HuBERT model. For more details, please refer to our [`paper`](https://arxiv.org/abs/2603.12046). 

---

## Requirements

To setup the environment, please refer to the official [`AV-HuBERT repository`](https://github.com/facebookresearch/av_hubert) with all the details. Once this is done, make sure to install the **shap** and **wandb** libraries: ```pip install shap wandb==0.15.12```. In addition to this, download the AV-HuBERT checkpoint we used in our manuscript. The ckpt we used is the Noise-Augmented AV-HuBERT Large model pre-trained on LRS3 + VoxCeleb2 (En) and finetuned on LRS3-433h. The ckpt name is `large_noise_pt_noise_ft_433h.pt`.  

## Compute the global A/V-SHAP Contributions.

To compute the A/V-SHAP contributions, run the command as below after cding into `av_hubert` folder: 

```Shell
python -B infer_s2s_shap.py --num-samples-shap [nums_samples_shap] --wandb-project [wandb_project] \
--exp-name [exp_name] --output-path [output_path] --shap-alg [shap_alg] --config-dir ./conf/ \
--config-name s2s_decode dataset.gen_subset=test +override.data=[path_to_data] +override.label_dir=[path_to_data] \
common_eval.path=[path_to_ckpt] common_eval.results_path=[/path/to/experiment/decode/s2s/test] \
override.modalities=['audio','video'] common.user_dir=`pwd` generation.beam=1 \
+override.noise_wav=[path_to_noise] override.noise_prob=1 override.noise_snr=-10
```

 <details open>
  <summary><strong>Main Arguments</strong></summary>

- `num-samples-shap`: The number of coalitions to sample.
- `wandb-project`: Name of the wandb project to track the results.
- `exp-name`: The experiment name.
- `output-path`: The path to save the SHAP values for further analyses. This folder must be created beforehand!
- `shap-alg`: The algorithm from the shap library to compute the shapley matrix. Choices: [`sampling`, `permutation`].
- `override.data/override.label_dir`: The path to the test.{tsv,wrd} files.
- `common_eval.path`: The path to the AV-HuBERT checkpoint.
- `common_eval.results_path`: The decoding results will be saved at this path.
- `override.noise_wav`: The path to the folder containing noise manifest files ([path_to_noise]/{valid,test}.tsv).
- `override.noise_snr`: The SNR level of acoustic noise to test on. Drop `override.noise_prob` and `override.noise_snr` if you want to test in clean conditions.

</details>


---

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{cappellazzo2026ODrSHAPAV,
  title={Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition},
  author={Umberto, Cappellazzo and Stavros, Petridis and Maja, Pantic},
  journal={arXiv preprint arXiv:2603.12046},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- Our code relies on [AV-HuBERT](https://github.com/facebookresearch/av_hubert)

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://umbertocappellazzo.github.io/Dr-SHAP-AV/) and our [preprint](https://arxiv.org/abs/2603.12046)

---

