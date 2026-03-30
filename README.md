# ISRF (WWW'2026)

This is the official implementation of the WWW 2026 paper: **"Iterative Semantic Reasoning from Individual to Group Interests for Generative Recommendation with LLMs"**.

![ISRF.drawio](https://raw.githubusercontent.com/htired/ISRF/refs/heads/main/ISRF.drawio.png))

## :bookmark_tabs:Data preprocessing

:one:Download the metadata for **Toys**, **Beauty**, and **Sports** from the [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) dataset, and place the files in `data/raw/{dataset}`.

```sh
cd data/preparation
```

:two: Then use `meta2csv.ipynb` to generate the extracted attribute file `meta-{dataset}.csv`.

## :rainbow:Item feature and user preference generation

> :microphone:Note: To make related information generation more convenient, we use **deepseek-chat** as the default reasoning model. You can also deploy **deepseek-r1:14b** locally as the reasoning model. For example, local reasoning can be implemented as shown in `data/{dataset}/generation/local_reasoning`.. 

Enter the `data/{dataset}/Generation` directory.

```sh
cd data/{dataset}/generation
```

### :scroll:Item feature generation

##### **Step1:Generate positive item descriptions**

```sh
cd item_feature
```

:one: Prepare item prompts by using `step1_positive/Item_prompt_construction.ipynb` to construct prompts from the basic item attributes.

:two: Run the following command to generate positive item descriptions.

```sh
python generate_item_information.py \
	--system_prompt_file ./step1_positive/step1_item_positive.txt \
	--item_prompt_file ./step1_positive/step1_item_positive_prompt.json \
	--response_file ./step1_positive/positive_response.txt
```

:three: Run the `extract2sum.py` to obtain the summarization about the item.

```sh
python extract2sum.py \
	--input_file ./step1_positive/positive_response.txt \
	--output_file ./step1_positive/responses_positive_item.json
```

---

##### **Step2:Generate negative item descriptions**

① Run `step2_merge2prompt.py` to construct the base item attribute prompts along with the previously generated item summary descriptions.

```sh
python step2_merge2prompt.py \
  --products_file ./step1_positive/toys_products.json \
  --profiles_file ./step1_positive/responses_positive_item.json \
  --merged_output_file ./step2_negative/step2_item_negative.json \
  --prompt_output_file ./step2_negative/step2_item_negative_prompt.json
```

②​ Run the following command to generate negative item descriptions.

```sh
python generate_item_information.py \
	--system_prompt_file ./step2_negative/step2_item_negative.txt \
	--item_prompt_file ./step2_negative/step2_item_negative_prompt.json \
	--response_file ./step2_negative/negative_response.txt
```

③​ Run the `extract2sum.py` to obtain the summarization about the item.

```sh
python extract2sum.py \
	--input_file ./step2_negative/negative_response.txt \
	--output_file ./step2_negative/responses_negative_item.json
```

---

##### **Step3:Generate item feature descriptions**

:one:Run `step3_merge2prompt.py` to construct the base item attribute prompts, along with the previously generated positive and negative summary descriptions of items. 

```sh
python step3_merge2prompt.py \
  --products_file ./step1_positive/toys_products.json \
  --profiles_file ./step1_positive/responses_positive_item.json \
  --negative_file ./step2_negative/responses_negative_item.json \
  --merged_output_file ./step3_feature/step3_item_feature.json \
  --prompt_output_file ./step3_feature/step3_item_feature_prompt.json
```

:two: Run the following command to generate negative item descriptions.

```sh
python generate_item_information.py \
	--system_prompt_file ./step3_feature/step3_item_feature.txt \
	--item_prompt_file ./step3_feature/step3_item_feature_prompt.json \
	--response_file ./step3_feature/feature_response.txt
```

:three: Run the `extract2sum.py` to obtain the feature description about the item.

```sh
python extract2sum.py \
	--input_file ./step3_feature/feature_response.txt \
	--output_file ./step3_feature/responses_feature_item.json
```

### :dart:User preference generation

```sh
cd user_preference
```

##### **Step1:Generate positive item descriptions**

① Run the following command to prepare the positive user input prompts.

```sh
python user_base_prompt.py \
  --products_file ../item_feature/step1_positive/toys_products.json \
  --sequential_txt ../../sequential.txt \
  --top_k 20 \
  --user_product_output ./step1_positive/step1_user_positive.json \
  --prompt_output ./step1_positive/step1_user_positive_prompt.json
```

② Run the following command to generate positive user descriptions.

```sh
python generate_user_information.py \
	--system_prompt_file ./step1_positive/step1_user_positive.txt \
	--item_prompt_file ./step1_positive/step1_user_posotive_prompt.json \
	--response_file ./step1_positive/positive_response.txt
```

③ Run the `extract2sum.py` to obtain the positive summarization about the user.

```sh
python extract2sum.py \
	--input_file ./step3_feature/positve_response.txt \
	--output_file ./step3_feature/responses_positive_user.json
```

---

##### **Step2:Generate negative user descriptions**

:one:Run `step2_merge2prompt.py` to construct the base user input prompts along with the previously generated positive user descriptions. 

```sh
python step2_merge2prompt.py \
  --user_input ./step1_positive/step1_user_positive.json \
  --responses_input ./step1_positive/responses_positive_user.json \
  --merged_output_file ./step2_negative/step2_user_negative.json \
  --prompt_output_file ./step2_negative/step2_user_negative_prompt.json
```

:two: Run the following command to generate negative item descriptions.

```sh
python generate_user_information.py \
	--system_prompt_file ./step2_negative/step2_user_negative.txt \
	--item_prompt_file ./step2_negative/step2_user_negative_prompt.json \
	--response_file ./step2_negative/negative_response.txt
```

:three: Run the `extract2sum.py` to obtain the summarization about the item.

```sh
python extract2sum.py \
	--input_file ./step2_negative/negative_response.txt \
	--output_file ./step2_negative/responses_negative_user.json
```

---

##### **Step3:Generate user preference descriptions**

① Run `step3_merge2prompt.py` to construct the base user input prompts along with the previously generated positive and negative user descriptions.。

```sh
python step3_merge2prompt.py \
  --user_input ./step2_negative/step2_user_negative.json \
  --responses_input ./step2_negative/responses_negative_user.json \
  --merged_output_file ./step3_preference/step2_user_preference.json \
  --prompt_output_file ./step3_preference/step2_user_preference_prompt.json
```

② Run the following command to generate negative item descriptions.

```sh
python generate_user_information.py \
	--system_prompt_file ./step3_preference/step3_user_preference.txt \
	--item_prompt_file ./step3_preference/step3_user_preference_prompt.json \
	--response_file ./step3_preference/preference_response.txt
```

③ Run the `extract2sum.py` to obtain the feature description about the item.

```sh
python extract2sum.py \
	--input_file ./step3_preference/preference_response.txt \
	--output_file ./step3_preference/responses_preference_user.json
```

### :page_facing_up: Embedding Construction

1. Clone the EasyRec repository

```sh
git clone https://github.com/HKUDS/EasyRec.git
```

2. Copy `generation/encode_easyrec_ISRF.py` into the `encode_easyrec` directory of the EasyRec repository.

3. Create a `data_ISRF` directory and place the JSON files that need to be encoded inside the corresponding dataset folder.

   Then run the following command:

   ```sh
   python encode_easyrec_ISRF.py --model baseline_embedders/easyrec-roberta-large --cuda 0 --datasets toys --output_dir ./data_ISRF --input_filename responses_feature_item.json --batch_size 128 --max_length 512 --output_filename item_feature_embedding_easyrec.pkl
   ```

4. Move the generated files (e.g., `item_feature_embedding_easyrec.pkl` or `user_preference_embedding_easyrec.pkl`) into `data/{dataset}/semantic`

----

```sh
cd data/{dataset}/semantic
```

5.  Follow the instructions in `retrieval_users.ipynb` to generate the top-K similar users pickle file: `user_preference_{}.pkl`
6.  Follow the instructions in `pca.ipynb` to generate the item embedding pickle file: `pca_item_feature.pkl`

## :art:Note (Coming Soon)

For ease of reproduction, we provide the processed datasets and model checkpoints on [Google Drive](https://drive.google.com/drive/folders/1uR7unI9q_3ke994Kp-G5mvaaLJpqJl0Q?usp=drive_link).

## :airplane:Examples to run the codes

:one:Pretraining ({dataset}: beauty, sports, and toys.)

```bash
python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/
```

:two:Inference ({dataset}: beauty, sports, and toys.) 

```bash
python SR.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
python DR.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
```

## :pray:Acknowledgement

Code reference [ELMRec](https://github.com/WangXFng/ELMRec/tree/main), [EasyRec](https://github.com/WangXFng/ELMRec/tree/main). 

