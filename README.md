# Bitcoin

In this repository, I tried of calculate nonce for POW from BlockChain header fields using Transformers and Hugginface Library. 

Result:
1. My transformer model over fitted
2. Bert - No appreciable learning	
Possible reasons :
1. There were only 100K datapoints and Transformers are notoriously data hungry
2. Bitcoin uses double AES.

ToDo:
1. Train DNN to generate hash. Google's paper has shown that DNNs are lot faster than tranditional algos. Using DNN to calculate hash may give you added advantage.  

