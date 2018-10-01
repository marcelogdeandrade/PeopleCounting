from codebook import Codebook

#codebooks = Codebook.train_codebooks()
#Codebook.save_codebooks(codebooks)

codebooks = Codebook.load_codebooks()
Codebook.test_codebooks(codebooks)