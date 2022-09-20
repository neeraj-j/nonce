# Bitcoin

In this repository, I tried of calculate nonce for POW from BlockChain header fields using Transformers and Hugginface Library. 

Result:
	This experiment didnt work. 
Possible reasons of failure:
1. There were only 100K datapoints and Transformers are notoriously data hungry
2. Bitcoin uses double AES.

ToDo:
1. Train DNN to generate hash. Google paper has shown that DNNs are lot faster than tranditional algos. Using DNN for calculatin hash may give you added advantage.  

