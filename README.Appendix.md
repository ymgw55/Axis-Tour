
# Appendix

## Preliminary

### Access to the embeddings used in our paper
Instead of recomputing the embeddings, you can access the embeddings used in the paper through the following links. Note that sign flip was not applied to the ICA-transformed embeddings to ensure that the skewness of the axes remains positive.

#### Raw embeddings
- [word2vec (Google Drive)](https://drive.google.com/file/d/16qHLV6iw0XPRZUI4wuDmJfzeVWldKsT-/view?usp=sharing)
- [BERT (Google Drive)](https://drive.google.com/file/d/16qHLV6iw0XPRZUI4wuDmJfzeVWldKsT-/view?usp=drive_link)

Place the downloaded files under the directory `output/raw_embeddings` as shown below:
```bash
$ ls output/raw_embeddings/
raw_bert.pkl  raw_glove.pkl  raw_word2vec.pkl
```

#### PCA-transformed and ICA-transformed embeddings
- [word2vec (Google Drive)](https://drive.google.com/file/d/14HyYRWAUuafs_MLP6C-Ad1X0A-52cEDR/view?usp=sharing)
- [BERT (Google Drive)](https://drive.google.com/file/d/1BDPenaBpwz0ceUc5aobrqEVzHfq7Dxri/view?usp=sharing)

Place the downloaded files under the directory `output/pca_ica_embeddings/` as shown below:
```bash
$ ls output/pca_ica_embeddings/
pca_ica_bert.pkl  pca_ica_glove.pkl  pca_ica_word2vec.pkl
```

#### Axis Tour embedidngs
- [GloVe (Google Drive)](https://drive.google.com/drive/folders/1uyViOqkVnzxxWRLzYBrD3dOQxOQbvIBb?usp=drive_link) for $k=1,10,1000$
- [word2vec (Google Drive)](https://drive.google.com/file/d/1zFMNILBZaLUYXROOzkWIc-NOqdFSd-d7/view?usp=sharing) for $k=100$
- [BERT (Google Drive)](https://drive.google.com/file/d/1Fy1TjTw4sHaAacVxLzb7NvVCTJis7eAQ/view?usp=sharing) for $k=100$

Place the downloaded files under the directory `output/axistour_embeddings/` as shown below:
```bash
$ ls output/axistour_embeddings/
axistour_top1000_glove.pkl  axistour_top100_bert.pkl  axistour_top100_glove.pkl  axistour_top100_word2vec.pkl  axistour_top10_glove.pkl  axistour_top1_glove.pkl
```

#### TICA9-transformed and TICA75-transformed GloVe embeddings
- [TICA9 and TICA75 (Google Drive)](https://drive.google.com/drive/folders/1ZfqHaco59SBijdZNhg19TOy-k3Nwfm3T?usp=sharing)

Place the downloaded files under the directory `output/tica_embeddings/` as shown below:
```bash
$ ls output/tica_embeddings/
tica_width75_glove.pkl  tica_width9_glove.pkl 
```


### Download datasets to compute BERT embeddings for reproducibility experiments (in necessary)

#### One Billion Word Benchmark [3] for BERT embeddings

Download it from the following link:

- https://www.statmt.org/lm-benchmark/

Please place the data as in `data/1-billion-word-language-modeling-benchmark-r13output/`.

### Download word2vec for reproducibility experiments (if necessary)

Make the `data/embeddings/word2vec` directory.

```bash
mkdir -p data/embeddings/word2vec
```

Then download it from the following link:

[GoogleNews-vectors-negative300.bin.gz (Google Cloud)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=drive_link&resourcekey=0-wjGZdNAUop6WykTtMip30g)


Please place the data as in `data/embeddings/word2vec/GoogleNews-vectors-negative300.bin`.

---

## Code

### Save embeddings for reproducibility experiments

```bash
# word2vec
python save_raw_embeddings.py --emb_type word2vec
python save_pca_and_ica_embeddings.py --emb_type word2vec
python save_axistour_embeddings.py --emb_type word2vec --topk 100

# bert
python save_raw_embeddings.py --emb_type bert
python save_pca_and_ica_embeddings.py --emb_type bert
python save_axistour_embeddings.py --emb_type bert --topk 100
```

### Scatterplots

If you are not using `adjustText==1.0.4`, you may need to manually adjust the position of the text.

#### glove

```bash
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 23 --length 9
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 101 --length 9
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 237 --length 9
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 23rd axis to the 31st axis</th>
  <th style="width: 33%;">(b) The 101st axis to the 109th axis</th>
  <th style="width: 33%;">(c) The 237th axis to the 245th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left23_length9.png" alt="fig. 6a"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left101_length9.png" alt="fig. 6b"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left237_length9.png" alt="fig. 6c"></td>
 </tr>
</table>

#### word2vec

```bash
python make_scatterplots.py --emb_type word2vec --topk 100 --left_axis_index 50 --length 10
python make_scatterplots.py --emb_type word2vec --topk 100 --left_axis_index 150 --length 10
python make_scatterplots.py --emb_type word2vec --topk 100 --left_axis_index 250 --length 10
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 50th axis to 59th axis</th>
  <th style="width: 33%;">(b) The 150th axis to 159th axis</th>
  <th style="width: 33%;">(c) The 250th axis to 259th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots/scatterplot_word2vec_top100_left50_length10.png" alt="fig. 6a"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_word2vec_top100_left150_length10.png" alt="fig. 6b"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_word2vec_top100_left250_length10.png" alt="fig. 6c"></td>
 </tr>
</table>

#### BERT

```bash
python make_scatterplots.py --emb_type bert --topk 100 --left_axis_index 50 --length 10
python make_scatterplots.py --emb_type bert --topk 100 --left_axis_index 150 --length 10
python make_scatterplots.py --emb_type bert --topk 100 --left_axis_index 250 --length 10
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 50th axis to 59th axis</th>
  <th style="width: 33%;">(b) The 150th axis to 159th axis</th>
  <th style="width: 33%;">(c) The 250th axis to 259th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots/scatterplot_bert_top100_left50_length10.png" alt="fig. 6a"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_bert_top100_left150_length10.png" alt="fig. 6b"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_bert_top100_left250_length10.png" alt="fig. 6c"></td>
 </tr>
</table>

### 3D projection


```bash
python make_3d_figure.py --emb_type glove --topk 100 --start_axis_index 89
```

<div align="center">
<img src="images_camera_ready/3d_figures/3d_figure_glove_top100_axis89_trimmed.png" alt="fig.7" width="50%">
</div>


### Comparing $k$
```bash
python make_comparing_k.py --emb_type glove
```
<div align="center">
<img src="images_camera_ready/comparing_k/comparing_k_glove.png" alt="fig.9" width="50%">
</div>

### Dimensionality reduction

#### Comparison of $\alpha$

```bash
python make_dimred_figure.py --emb_type glove --fig_type alpha
```
<div align="center">
<img src="images_camera_ready/dimred/dimred_glove_alpha.png" alt="fig.8" width="95%">
</div>

#### Comparison of $k$

```bash
python make_dimred_figure.py --emb_type glove --fig_type topk
```
<div align="center">
<img src="images_camera_ready/dimred/dimred_glove_topk.png" alt="fig.10" width="95%">
</div>

#### Skewness Sort Projection and Random Order Projection

```bash
python make_dimred_figure.py --emb_type glove --fig_type projection
```
<div align="center">
<img src="images_camera_ready/dimred/dimred_glove_projection.png" alt="fig.10" width="95%">
</div>

### TICA

#### Save embeddings for reproducibility experiments

```bash
python save_tica_embeddings.py --emb_type glove --width 9
python save_tica_embeddings.py --emb_type glove --width 75
```

#### Scatterplots

If you are not using `adjustText==1.0.4`, you may need to manually adjust the position of the text.

##### TICA9

```bash
python make_scatterplots_tica.py --emb_type glove --width 9 --left_axis_index 50 --length 10
python make_scatterplots_tica.py --emb_type glove --width 9 --left_axis_index 150 --length 10
python make_scatterplots_tica.py --emb_type glove --width 9 --left_axis_index 250 --length 10
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 50th axis to the 59th axis</th>
  <th style="width: 33%;">(b) The 150st axis to the 159th axis</th>
  <th style="width: 33%;">(c) The 250th axis to the 259th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width9_left50_length10.png" alt="fig. 20a"></td>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width9_left150_length10.png" alt="fig. 20b"></td>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width9_left250_length10.png" alt="fig. 20c"></td>
 </tr>
</table>

##### TICA75

```bash
python make_scatterplots_tica.py --emb_type glove --width 75 --left_axis_index 50 --length 10
python make_scatterplots_tica.py --emb_type glove --width 75 --left_axis_index 150 --length 10
python make_scatterplots_tica.py --emb_type glove --width 75 --left_axis_index 250 --length 10
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 50th axis to the 59th axis</th>
  <th style="width: 33%;">(b) The 150st axis to the 159th axis</th>
  <th style="width: 33%;">(c) The 250th axis to the 259th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width75_left50_length10.png" alt="fig. 20a"></td>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width75_left150_length10.png" alt="fig. 20b"></td>
  <td><img src="images_camera_ready/scatterplots_tica/scatterplot_glove_width75_left250_length10.png" alt="fig. 20c"></td>
 </tr>
</table>

##### glove

```bash
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 50 --length 10
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 150 --length 10
python make_scatterplots.py --emb_type glove --topk 100 --left_axis_index 250 --length 10
```

<table>
 <tr>
  <th style="width: 33%;">(a) The 50th axis to the 59th axis</th>
  <th style="width: 33%;">(b) The 150st axis to the 159th axis</th>
  <th style="width: 33%;">(c) The 250th axis to the 259th axis</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left50_length10.png" alt="fig. 20a"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left150_length10.png" alt="fig. 20b"></td>
  <td><img src="images_camera_ready/scatterplots/scatterplot_glove_top100_left250_length10.png" alt="fig. 20c"></td>
 </tr>
</table>


#### Histograms and scatterplot of cossim

```bash
python make_cossim_histogram_and_scatterplot_tica.py --emb_type glove
```

<table>
 <tr>
  <th style="width: 50%;">Figure 21</th>
  <th style="width: 50%;">Figure 22</th>
 </tr>
 <tr>
  <td><img src="images_camera_ready/tica/tica_histogram_glove_top100.png" alt="fig. 21"></td>
  <td><img src="images_camera_ready/tica/tica_scatter_glove_top100.png" alt="fig. 22"></td>
 </tr>
</table>

#### Avg. $d_I$ and Avg. $c_I$

```bash
# Axis Tour
python eval_avg_d_I_and_avg_c_I.py
# TICA
python eval_avg_d_I_and_avg_c_I_tica.py --width 9
python eval_avg_d_I_and_avg_c_I_tica.py --width 75
```

#### Histograms of higher-order correlation

```bash
python make_higher_order_histogram.py --emb_type glove
```

<div align="center">
<img src="images_camera_ready/tica/tica_ho_histogram_glove_top100.png" alt="fig.24" width="50%">
</div>

#### Dimensionality reduction

```bash
python make_dimred_figure.py --emb_type glove --fig_type tica
```
<div align="center">
<img src="images_camera_ready/dimred/dimred_glove_tica.png" alt="fig.23" width="95%">
</div>