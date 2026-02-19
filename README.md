## CROSS-MODAL FOUNDATION MODEL FUSION FOR KEYSTROKE INFERENCE ON WEARABLE RINGS

This is a description of the scripts that I have so far. I will be making a proper data loader soon. In the mean time, you can use and modify these scripts to explore the dataset. 

FYI, these scripts were primarily AI generated as a way for me to quickly explore the data. I am unaware of any bugs since I have not looked too carefully at the code.

## Scripts

```explore_data.py``` 
calculates general statistics about the dataset, like how much typing data is in each session, how many keystrokes were recorded, keystroke distributions, etc.

```regenerate_text.py``` 
generates coherent text from the noisy keystroke sequences

```stress_test.py``` 
tests that the gpu set up has enough VRAM to handle a forward and backward pass of GPT2

```gpu_test.py``` 
checks GPU availability
