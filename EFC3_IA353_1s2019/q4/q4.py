from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.generate()

file_path = "domcasmurro.txt"
textgen.reset()
textgen.train_from_file(file_path, new_model=True, num_epochs=20,gen_epochs=10, word_level=True)
textgen.generate(interactive=True, top_n=5)
textgen.save('domcasmurro.hdf5')
# fim
