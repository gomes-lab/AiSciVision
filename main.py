import json
import os
import sys
import time

from PIL import Image
from sklearn.metrics import roc_auc_score

import aiSciVision
import config
import dataloaders.datasets
import embeddingModel
import lmm
import promptSchema
import utils
import visualRAG

# 1. Parse all experimental settings
args = utils.setup_experiment()

# 2. Set random seed for reproducibility
utils.seed_everything(args.seed)

# 3. Load the dataset
train_dataloader, val_dataloder, test_dataloader = dataloaders.datasets.get_dataloaders(
    args.data_root_dir,
    args.dataset_name,
    args.batch_size,
    args.percent_labeled,
    args.test_order_path,
    args.use_val_dataset,
)

# 4. Load the prompt schema
prompt_schema = promptSchema.get_prompt_schema(
    args.prompt_schema_name, args.tools, args.clip_model_path, args.num_tool_rounds
)

# 5. Setup the embedding model, and visual RAG
embedding_model_init_kwargs = {}
if args.embedding_model_name in ["clip", "clip_vision"]:
    embedding_model_init_kwargs = dict(clip_model_config=config.clip_model_config)
emb_model = embeddingModel.get_embedding_model(args.embedding_model_name, embedding_model_init_kwargs)
vis_rag = visualRAG.get_visual_rag(args.rag_type, train_dataloader, emb_model)

# 6. Setup the LMM
lmm_model = lmm.get_lmm(args.lmm_name, seed=args.seed)
# Get the system fingerprint from the LMM model
lmm_system_fingerprint = lmm_model.get_system_fingerprint()

# Write the system fingerprint to a JSON file
fingerprint_file_path = os.path.join(args.log_folder, "lmm_system_fingerprint.json")
with open(fingerprint_file_path, "w") as f:
    json.dump({"system_fingerprint": lmm_system_fingerprint}, f, indent=4)
print(f"LMM system fingerprint saved to {fingerprint_file_path}")

# 7. Iterate through the test set
lmm_preds = []
supervised_preds = []
gts = []
ids = []
print(f"Number of images in train dataset: {len(train_dataloader.dataset)}")
print(f"Number of images in test dataset: {len(test_dataloader.dataset)}")
total_looked_at = 0

start_time = time.time()
total_time = 0
total_batches = 0
total_images = 0

# Check if sum_stats.json exists and load the processed examples
sum_stats_path = os.path.join(args.log_folder, "sum_stats.json")
if os.path.exists(sum_stats_path):
    with open(sum_stats_path, "r") as f:
        sum_stats = json.load(f)
    ids = sum_stats["ids"]
    lmm_preds = sum_stats["lmm_preds"]
    supervised_preds = sum_stats["supervised_preds"]
    gts = sum_stats["ground_truths"]
    total_looked_at = sum_stats["total_looked_at"]
    total_time = sum_stats["total_run_time"]
    total_images = sum_stats["total_images"]
    start_time = time.time() - total_time
else:
    sum_stats = {}

if total_looked_at >= args.number_of_test_samples:
    print(f"Already processed {total_looked_at} samples. No further processing needed.")
    # Exit the script as no further processing is needed
    print("Exiting the script as requested.")
    sys.exit(0)

# Calculate the starting batch and image index if we've already processed some images
if total_looked_at > 0:
    batch_size = test_dataloader.batch_size
    starting_i = total_looked_at // batch_size
    starting_j = total_looked_at % batch_size

    print(f"Resuming from batch {starting_i}, image {starting_j}")
else:
    starting_i = 0
    starting_j = 0

for i, (no_transformed_images, transformed_images, labels, metadatas) in enumerate(test_dataloader):
    if i < starting_i:
        continue
    batch_start_time = time.time()

    for j in range(no_transformed_images.shape[0]):
        if i == starting_i and j < starting_j:
            continue
        image_start_time = time.time()

        no_transformed_image = no_transformed_images[j]
        transformed_image = transformed_images[j]
        label = labels[j]

        # Initialize log for this image
        image_log = {
            "batch": i,
            "image": j,
            "label": label.item(),
            "tools_used": [],
            "lmm_prediction": None,
            "supervised_prediction": None,
            "probability": None,
        }

        # 8. Process image with agent
        aisci_vision = aiSciVision.AiSciVision(vis_rag, prompt_schema)

        # 8a. First set system prompt
        aisci_vision.set_system_prompt()

        # 8b. Now get initial prompt from visual RAG
        initial_prompts = aisci_vision.get_initial_prompts(no_transformed_image.unsqueeze(0))

        # 8c. Now we iterate through the initial prompts and send them to the LMM
        for text, image in initial_prompts:
            # 8c.1 Update conversation history
            aisci_vision.update_conversation(role="user", message=text, image=image)

            # 8c.2 Send prompt to LMM
            lmm_response = lmm_model.process_conversation(aisci_vision.conversation)

            # 8c.3 Update conversation history
            aisci_vision.update_conversation(role="assistant", message=lmm_response, image=None)

        # 8d. Now we iterate through the tool rounds
        # Have initial prompt with tool explanation
        if args.num_tool_rounds > 0:
            tool_prompt = prompt_schema.get_tool_usage_prompt(round_num=0)
            aisci_vision.update_conversation(role="user", message=tool_prompt, image=None)

            # Extract latitude and longitude for aquaculture dataset, otherwise set to None
            if args.dataset_name == "aquaculture":
                image_metadata = {
                    "original_lat": metadatas["lat"][j],
                    "original_lon": metadatas["lon"][j],
                    "current_lat": metadatas["lat"][j],
                    "current_lon": metadatas["lon"][j],
                    "original_zoom": 19,
                    "zoom": 19,
                    "relative_to_original": False,  # Set this to True if you want actions relative to the original image
                }
                image_log["latitude"] = metadatas["lat"][j].item()
                image_log["longitude"] = metadatas["lon"][j].item()
            else:
                image_metadata = {}

        # Now iterate through number of allowed rounds
        for round_num in range(args.num_tool_rounds):
            # 8d.3 Send prompt to LMM
            lmm_response = lmm_model.process_conversation(aisci_vision.conversation)

            # 8d.4 Update conversation history
            aisci_vision.update_conversation(role="assistant", message=lmm_response, image=None)

            # 8d.5 Use tool
            try:
                tool_name = lmm_response.split("[")[1].split("]")[0]
            except IndexError as e:
                print(
                    f"! ERROR: LMM response to tool not correct. Could not extract tool name from response:\n'''\n{lmm_response}\n'''"
                )
                print("---- Image Log ----")
                print(image_log)
                raise e

            if tool_name.lower() == "finished":
                image_log["tools_used"].append({"round": round_num, "tool": "FINISHED"})
                break

            tool_input_image = Image.fromarray((no_transformed_image.squeeze().permute(1, 2, 0) * 255.0).byte().numpy())
            if args.dataset_name == "aquaculture":
                tool_response, tool_image, updated_image_metadata = prompt_schema.use_tool(
                    tool_name, tool_input_image, round_num + 1, image_metadata
                )
                image_metadata = updated_image_metadata
            else:
                tool_response, tool_image, updated_image_metadata = prompt_schema.use_tool(
                    tool_name, tool_input_image, round_num + 1, image_metadata
                )

            # 8d.6 Update conversation history
            aisci_vision.update_conversation(role="user", message=tool_response, image=tool_image)

            image_log["tools_used"].append({"round": round_num, "tool": tool_name})

        # 8e. Process the final answer
        # 8e.1 Get the final answer
        final_prompt = aisci_vision.get_final_prompt()
        aisci_vision.update_conversation(role="user", message=final_prompt, image=None)

        final_answer = lmm_model.process_conversation(aisci_vision.conversation)
        aisci_vision.update_conversation(role="assistant", message=final_answer, image=None)

        # 8e.2 Parse the final answer
        parsed_final_answer = aisci_vision.parse_final_answer(final_answer)
        image_log["lmm_prediction"] = parsed_final_answer["class"]
        image_log["lmm_probability"] = parsed_final_answer["probability"]

        # 8e.3 Evaluate the final answer
        utils.evaluate_final_answer(parsed_final_answer["class"], label)

        lmm_preds.append({"class": parsed_final_answer["class"], "probability": parsed_final_answer["probability"]})
        gts.append(label.item())

        ### Aside: Get predictions from supervised model ###
        # 8f. Get predictions from supervised model tool (LMM is not called here)
        tool_input_image = Image.fromarray((no_transformed_image.squeeze().permute(1, 2, 0) * 255.0).byte().numpy())
        tool_response_text, _, _ = prompt_schema.get_supervised_tool_probability(
            tool_input_image, image_metadata if "image_metadata" in locals() else {}
        )

        # Extract probability from tool response
        probability_str = tool_response_text.split("image is")[1].split("%")[0].strip()
        probability = float(probability_str)

        # Determine supervised prediction
        supervised_prediction = 1 if probability >= 50 else 0
        supervised_preds.append({"class": supervised_prediction, "probability": probability})
        image_log["supervised_prediction"] = supervised_prediction
        image_log["supervised_probability"] = probability
        ### End Aside ###

        # 8g. Save conversation text to image log
        image_log["conversation"] = []

        # Save images from conversation to separate folder
        image_folder = os.path.join(args.log_folder, "conversation_images")
        os.makedirs(image_folder, exist_ok=True)
        for idx, msg in enumerate(aisci_vision.conversation):
            conversation_entry = {"role": msg["role"], "content": msg["message"][0]}
            if msg["message"][1] is not None:
                image_path = os.path.join(image_folder, f"image_batch_{i}_im_{j}_id_{total_looked_at}_msg_{idx}.png")
                msg["message"][1].save(image_path)
                conversation_entry["image_path"] = image_path
            image_log["conversation"].append(conversation_entry)

        print(f"Supervised prediction: {supervised_prediction}, LMM prediction: {parsed_final_answer['class']}")
        print(f"Supervised probability: {probability}, LMM probability: {parsed_final_answer['probability']}")
        print(f"Label: {label}")

        total_looked_at += 1
        ids.append(total_looked_at)

        # 8h. Save the conversation to a PDF, including true class, LMM prediction, and supervised prediction
        pdf_dir = os.path.join(args.log_folder, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        utils.create_conversation_pdf(
            aisci_vision.conversation,
            os.path.join(
                pdf_dir,
                f"conversation_batch_{i}_im_{j}_id_{total_looked_at}_true{label}_lmm{parsed_final_answer['class']}_sup{supervised_prediction}.pdf",
            ),
            true_class=label,
            lmm_prediction=parsed_final_answer["class"],
            lmm_probability=parsed_final_answer["probability"],
            supervised_prediction=supervised_prediction,
            supervised_probability=probability,
        )

        # Save image log to JSON
        image_log_dir = os.path.join(args.log_folder, "image_logs")
        os.makedirs(image_log_dir, exist_ok=True)
        with open(os.path.join(image_log_dir, f"image_log_batch_{i}_im_{j}_id_{total_looked_at}.json"), "w") as f:
            json.dump(image_log, f, indent=4)

        total_images += 1
        image_time = time.time() - image_start_time
        total_time += image_time
        print(f"Image {j} in batch {i} took {image_time:.2f} seconds")

        if total_looked_at >= args.number_of_test_samples:
            break

    if total_looked_at >= args.number_of_test_samples:
        break


# 9. Compute and print out accuracy, f1, precision, and recall of lmm and supervised model
lmm_accuracy = utils.compute_accuracy([pred["class"] for pred in lmm_preds], gts)
supervised_accuracy = utils.compute_accuracy([pred["class"] for pred in supervised_preds], gts)
lmm_f1 = utils.compute_f1([pred["class"] for pred in lmm_preds], gts)
supervised_f1 = utils.compute_f1([pred["class"] for pred in supervised_preds], gts)
lmm_precision = utils.compute_precision([pred["class"] for pred in lmm_preds], gts)
lmm_recall = utils.compute_recall([pred["class"] for pred in lmm_preds], gts)
supervised_precision = utils.compute_precision([pred["class"] for pred in supervised_preds], gts)
supervised_recall = utils.compute_recall([pred["class"] for pred in supervised_preds], gts)

# Calculate ROC AUC scores
# Handle the case where there's only one class for ROC AUC calculation
if len(set(gts)) == 1:
    lmm_roc_auc = -1
    supervised_roc_auc = -1
else:
    lmm_roc_auc = roc_auc_score(gts, [pred["probability"] for pred in lmm_preds])
    supervised_roc_auc = roc_auc_score(gts, [pred["probability"] for pred in supervised_preds])

print([pred["class"] for pred in lmm_preds])
print([pred["class"] for pred in supervised_preds])
print(gts)
print(f"Total Looked at {total_looked_at}")
print(f"LMM accuracy: {lmm_accuracy}, Supervised accuracy: {supervised_accuracy}")
print(f"LMM f1: {lmm_f1}, Supervised f1: {supervised_f1}")
print(f"LMM precision: {lmm_precision}, LMM recall: {lmm_recall}")
print(f"Supervised precision: {supervised_precision}, Supervised recall: {supervised_recall}")
print(f"LMM ROC AUC: {lmm_roc_auc}, Supervised ROC AUC: {supervised_roc_auc}")

end_time = time.time()
total_run_time = end_time - start_time
avg_time_per_image = total_time / total_images if total_images > 0 else 0

print(f"Total run time: {total_run_time:.2f} seconds")
print(f"Average time per image: {avg_time_per_image:.2f} seconds")

# Save summary statistics
sum_stats = {
    "ids": ids,
    "lmm_preds": lmm_preds,
    "supervised_preds": supervised_preds,
    "ground_truths": gts,
    "total_looked_at": total_looked_at,
    "lmm_accuracy": lmm_accuracy,
    "supervised_accuracy": supervised_accuracy,
    "lmm_f1": lmm_f1,
    "supervised_f1": supervised_f1,
    "lmm_precision": lmm_precision,
    "lmm_recall": lmm_recall,
    "supervised_precision": supervised_precision,
    "supervised_recall": supervised_recall,
    "lmm_roc_auc": lmm_roc_auc,
    "supervised_roc_auc": supervised_roc_auc,
    "total_run_time": total_run_time,
    "avg_time_per_image": avg_time_per_image,
    "total_images": total_images,
}
with open(os.path.join(args.log_folder, "sum_stats.json"), "w") as f:
    json.dump(sum_stats, f, indent=4, sort_keys=True)

# Print summary statistics after the experiment
print("\n--- Experiment Summary ---")
print(f"Total images processed: {total_images}")
print(f"Total run time: {total_run_time:.2f} seconds")
print(f"Average time per image: {avg_time_per_image:.2f} seconds")
print(f"\nFinal LMM accuracy: {lmm_accuracy:.4f}")
print(f"Final Supervised accuracy: {supervised_accuracy:.4f}")
print(f"Final LMM F1 score: {lmm_f1:.4f}")
print(f"Final Supervised F1 score: {supervised_f1:.4f}")
print(f"Final LMM Precision: {lmm_precision:.4f}")
print(f"Final LMM Recall: {lmm_recall:.4f}")
print(f"Final Supervised Precision: {supervised_precision:.4f}")
print(f"Final Supervised Recall: {supervised_recall:.4f}")

print("\nExperiment completed successfully!")
