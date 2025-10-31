# Hierarchical Fingerprinting (HF)

This project is based on the [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library).

## Running the HF Model

### Main Steps

1. **Download Dataset**
   - Download the dataset from [this link](https://drive.google.com/file/d/1Ix1ErnEEocpELHV483SKTcH_VV-pZesK/view?usp=sharing) and unzip it to the desired location.

2. **Prepare the Data and Configure Scripts**
   
   - Split the downloaded dataset into training, validation, and test sets according to the preset proportions.
   - Before running the scripts, open the relevant `.sh` files in the `WFlib/scripts` directory and change the dataset and checkpoint paths to your local paths.
   
3. **Run HF Script**
   
   - Navigate to the WFlib directory and execute the HF shell script:
     ```bash
     cd WFlib
     bash scripts/HF.sh
     ```

**More details will be available soon.**
