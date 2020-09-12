1. To run code for Binary Desicion Tree (Iris, Spambase, Mushroom Datasets):
	a. Open the terminal
	b. Switch directories to this folder
	c. Run the command --> python3 BinaryDecisionTree.py $datasetName $threshold
		For Iris Dataset: python3 BinaryDecisionTree.py Iris 0.05
						  python3 BinaryDecisionTree.py Iris 0.1
						  python3 BinaryDecisionTree.py Iris 0.15
						  python3 BinaryDecisionTree.py Iris 0.20
		For Spambase Dataset: python3 BinaryDecisionTree.py Spambase 0.05
							  python3 BinaryDecisionTree.py Spambase 0.10
							  python3 BinaryDecisionTree.py Spambase 0.15
							  python3 BinaryDecisionTree.py Spambase 0.20
							  python3 BinaryDecisionTree.py Spambase 0.25
		For Mushroom Dataset: python3 BinaryDecisionTree.py Mushroom 0.05
							  python3 BinaryDecisionTree.py Mushroom 0.10
							  python3 BinaryDecisionTree.py Mushroom 0.15
	d. Results will be printed to the console. 
	* Spambase dataset will take relatively long time to get the results.

2. To run code for Multiway Decision Tree (Mushroom Dataset):
	a. Open the terminal
	b. Switch directories to this folder
	c. Run the Command (python3 MultiwayDecisionTree.py $threshold):
			python3 MultiwayDecisionTree.py 0.05
			python3 MultiwayDecisionTree.py 0.10
			python3 MultiwayDecisionTree.py 0.15
			python3 MultiwayDecisionTree.py 0.20
	d. Results will be printed to the console.

3. To run code for Regression Tree (Housing Dataset):
	a. Open the terminal
	b. Switch directories to this folder
	c. Run the Command (python3 RegressionTree.py $threshold): 
			python3 RegressionTree.py 0.05
			python3 RegressionTree.py 0.10
			python3 RegressionTree.py 0.15
			python3 RegressionTree.py 0.20
	d. Results will be printed to the console.
