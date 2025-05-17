#!/bin/bash
set -e # Any command failing will cause the script to exit.

# --- User Configuration ---
# Set the project directory to the directory where this script is located.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the main configuration file
CONFIG_FILE="${PROJECT_DIR}/configs/432_config.yaml"

# Base directory for logs and checkpoints (relative to PROJECT_DIR)
LOGS_BASE_DIR="logs"

# Output directory for the overall acceptance test results summary
ACCEPTANCE_RESULTS_DIR="${PROJECT_DIR}/acceptance_summary_results"

# Name of the best model checkpoint file
BEST_MODEL_FILENAME="best_model.pdparams"
# Name of the metadata file associated with the best model
BEST_MODEL_METADATA_FILENAME="best_model.json"

# Recognition threshold for ArcFace acceptance (overrides config)
# Set to empty string to use value from config file
RECOGNITION_THRESHOLD="0.75" # Example threshold

# Set to "--use_gpu" to enable GPU, or "" to use CPU
USE_GPU_FLAG="--use_gpu" # Or comment out and use "" if you want to enforce CPU for acceptance


# --- Setup ---
echo "--- Starting Automated Acceptance Tests ---"

# Ensure acceptance results directory exists
mkdir -p "${ACCEPTANCE_RESULTS_DIR}"
RESULTS_CSV="${ACCEPTANCE_RESULTS_DIR}/acceptance_results_$(date +%Y%m%d-%H%M%S).csv"

# Write CSV header
echo "Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Feature_Library_Path,Status,Notes" > "${RESULTS_CSV}"

echo "Results will be logged to: ${RESULTS_CSV}"
echo ""

# Find all best model files
echo "Searching for best model files in ${LOGS_BASE_DIR}..."
# Use find to search for best_model.pdparams files
# -L follows symbolic links, -type f looks for files
FIND_CMD="find -L \"${PROJECT_DIR}/${LOGS_BASE_DIR}\" -type f -name \"${BEST_MODEL_FILENAME}\""
MODEL_FILES=$(eval ${FIND_CMD})

if [ -z "${MODEL_FILES}" ]; then
    echo "No best model files (${BEST_MODEL_FILENAME}) found in ${LOGS_BASE_DIR}. Exiting."
    exit 0
fi

echo "Found models to test:"
echo "${MODEL_FILES}"
echo ""

# --- Loop through found models ---
TOTAL_MODELS=$(echo "${MODEL_FILES}" | wc -l)
TESTED_COUNT=0

echo "${MODEL_FILES}" | while read MODEL_PATH; do
    TESTED_COUNT=$((TESTED_COUNT + 1))
    MODEL_DIR=$(dirname "${MODEL_PATH}") # Directory containing best_model.pdparams
    METADATA_PATH="${MODEL_DIR}/${BEST_MODEL_METADATA_FILENAME}"

    echo "--- Testing Model ${TESTED_COUNT}/${TOTAL_MODELS}: ${MODEL_PATH} ---"

    # Check if metadata file exists
    if [ ! -f "${METADATA_PATH}" ]; then
        echo "Error: Metadata file not found for ${MODEL_PATH}. Skipping."
        echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),Unknown,Unknown,Unknown,Unknown,Unknown,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",N/A,Skipped,\"Metadata file not found\"" >> "${RESULTS_CSV}"
        continue
    fi

    # Extract model type and loss type from metadata using grep+cut
    # Assuming loss_type and model_type are top-level keys in the JSON
    # Example line in JSON: "loss_type": "arcface",
    LOSS_TYPE=$(grep '"loss_type":' "${METADATA_PATH}" | head -n 1 | cut -d '"' -f 4)
    MODEL_TYPE=$(grep '"model_type":' "${METADATA_PATH}" | head -n 1 | cut -d '"' -f 4)
    TRAIN_ACC=$(grep '"last_eval_accuracy":' "${METADATA_PATH}" | head -n 1 | cut -d ':' -f 2 | cut -d ',' -f 1 | tr -d '[:space:]')
    EVAL_ACC=$(grep '"last_eval_accuracy":' "${METADATA_PATH}" | head -n 1 | cut -d ':' -f 2 | cut -d ',' -f 1 | tr -d '[:space:]') # Use last_eval_accuracy for both

    if [ -z "${LOSS_TYPE}" ] || [ -z "${MODEL_TYPE}" ]; then
        echo "Error: Could not extract model_type or loss_type from ${METADATA_PATH}. Skipping."
         echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE:-Unknown},${LOSS_TYPE:-Unknown},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Unknown,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",N/A,Skipped,\"Could not extract types from metadata\"" >> "${RESULTS_CSV}"
        continue
    fi

    echo "  Model Type: ${MODEL_TYPE}, Loss Type: ${LOSS_TYPE}"

    ACCEPTANCE_ACC="N/A" # Default acceptance accuracy
    FEATURE_LIB_PATH="N/A" # Default feature library path

    if [ "${LOSS_TYPE,,}" = "arcface" ]; then # Case-insensitive comparison
        echo "  Running ArcFace specific acceptance..."

        # Determine feature library path
        FEATURE_LIB_DIR="${MODEL_DIR}/feature_library"
        mkdir -p "${FEATURE_LIB_DIR}"
        FEATURE_LIB_NAME="face_library.pkl" # Match default in create_face_library.py config/code
        FEATURE_LIB_PATH="${FEATURE_LIB_DIR}/${FEATURE_LIB_NAME}"

        # Build feature library using training data and the current model
        echo "  Building feature library using training data..."
        # create_face_library.py loads config to find data paths and train list name
        python "${PROJECT_DIR}/create_face_library.py" --config_path "${CONFIG_FILE}" --model_path "${MODEL_PATH}" --output_library_path "${FEATURE_LIB_PATH}" ${USE_GPU_FLAG}
        CREATE_LIB_STATUS=$?

        if [ ${CREATE_LIB_STATUS} -ne 0 ]; then
            echo "  Error: Failed to create feature library. Skipping acceptance test for this model."
             echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Unknown,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",\"${FEATURE_LIB_PATH}\",Failed,\"Feature library creation failed\"" >> "${RESULTS_CSV}"
            continue
        fi
         if [ ! -f "${FEATURE_LIB_PATH}" ]; then
            echo "  Error: Feature library file ${FEATURE_LIB_PATH} not found after creation script finished. Skipping acceptance test."
             echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Unknown,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",\"${FEATURE_LIB_PATH}\",Failed,\"Feature library file not generated\"" >> "${RESULTS_CSV}"
            continue
        fi


        # Run acceptance test (recognition)
        echo "  Running acceptance test with feature library..."
        # acceptance_test.py expects --feature_library_path and --recognition_threshold
        # Use tee to capture output and display
        ACCEPTANCE_OUTPUT=$(python "${PROJECT_DIR}/acceptance_test.py" --config_path "${CONFIG_FILE}" --trained_model_path "${MODEL_PATH}" --feature_library_path "${FEATURE_LIB_PATH}" ${USE_GPU_FLAG} --recognition_threshold "${RECOGNITION_THRESHOLD}" 2>&1 | tee "${MODEL_DIR}/acceptance_test_arcface.log")
        ACCEPTANCE_STATUS=$?

        if [ ${ACCEPTANCE_STATUS} -ne 0 ]; then
            echo "  Error: Acceptance test script failed for ArcFace model."
            echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Failed,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",\"${FEATURE_LIB_PATH}\",Failed,\"Acceptance script returned non-zero exit code\"" >> "${RESULTS_CSV}"
            continue
        fi

        # Extract final accuracy from script output
        # Assuming the script prints "识别准确率 (阈值 > 0.xxxx): 0.yyyy"
        ACCEPTANCE_ACC=$(echo "${ACCEPTANCE_OUTPUT}" | grep "识别准确率 (阈值 >" | tail -n 1 | awk '{print $NF}')

        if [ -z "${ACCEPTANCE_ACC}" ]; then
             echo "  Warning: Could not extract acceptance accuracy from script output."
             ACCEPTANCE_ACC="ExtractionFailed"
        fi

        echo "  ArcFace Acceptance Accuracy: ${ACCEPTANCE_ACC}"
        STATUS="Success"
        NOTES=""

    elif [ "${LOSS_TYPE,,}" = "cross_entropy" ]; then # Case-insensitive comparison
        echo "  Running Cross-Entropy specific acceptance (classification)..."

        # Run acceptance test (classification)
        echo "  Running acceptance test..."
         # acceptance_test.py does classification if loss_type is cross_entropy
         # No feature_library_path or recognition_threshold needed
        ACCEPTANCE_OUTPUT=$(python "${PROJECT_DIR}/acceptance_test.py" --config_path "${CONFIG_FILE}" --trained_model_path "${MODEL_PATH}" ${USE_GPU_FLAG} 2>&1 | tee "${MODEL_DIR}/acceptance_test_ce.log")
        ACCEPTANCE_STATUS=$?

        if [ ${ACCEPTANCE_STATUS} -ne 0 ]; then
            echo "  Error: Acceptance test script failed for Cross-Entropy model."
            echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Failed,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",N/A,Failed,\"Acceptance script returned non-zero exit code\"" >> "${RESULTS_CSV}"
            continue
        fi

         # Extract final accuracy from script output
        # Assuming the script prints "分类准确率: 0.yyyy"
        ACCEPTANCE_ACC=$(echo "${ACCEPTANCE_OUTPUT}" | grep "分类准确率:" | tail -n 1 | awk '{print $NF}')

        if [ -z "${ACCEPTANCE_ACC}" ]; then
             echo "  Warning: Could not extract acceptance accuracy from script output."
             ACCEPTANCE_ACC="ExtractionFailed"
        fi

        echo "  Cross-Entropy Acceptance Accuracy: ${ACCEPTANCE_ACC}"
        STATUS="Success"
        NOTES=""

    else
        echo "  Error: Unsupported loss type '${LOSS_TYPE}'. Skipping acceptance test for this model."
        STATUS="Skipped"
        NOTES="Unsupported loss type"
        echo "$(basename "$(dirname "$(dirname "${MODEL_DIR}")")"),$(basename "$(dirname "${MODEL_DIR}")"),${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},Unknown,${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",N/A,Skipped,\"Unsupported loss type\"" >> "${RESULTS_CSV}"
        continue # Skip to next model
    fi

    # Log results to CSV
    # Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Feature_Library_Path,Status,Notes
    COMBO_NAME=$(basename "$(dirname "$(dirname "${MODEL_DIR}")")") # e.g., resnet_arcface_adamw_..._config
    TIMESTAMP=$(basename "$(dirname "${MODEL_DIR}")") # e.g., YYYYMMDD-HHMMSS

    echo "${COMBO_NAME},${TIMESTAMP},${MODEL_TYPE},${LOSS_TYPE},${TRAIN_ACC:-N/A},${EVAL_ACC:-N/A},${ACCEPTANCE_ACC},${RECOGNITION_THRESHOLD:-N/A},\"${MODEL_PATH}\",\"${FEATURE_LIB_PATH}\",${STATUS},\"${NOTES}\"" >> "${RESULTS_CSV}"
    echo "  Results logged to CSV."
    echo ""

done

echo "--- Automated Acceptance Tests Finished ---"
echo "Summary available in: ${RESULTS_CSV}"
