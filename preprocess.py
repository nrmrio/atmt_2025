import os
import pickle
import logging
import argparse
import sentencepiece as spm

from seq2seq.data.tokenizer import BPETokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess text data for training.")
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    # File paths
    parser.add_argument("--raw-data", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--dest-dir", type=str, default="./", help="Directory to save the processed data.")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save the trained tokenization model and vocab.")
    parser.add_argument("--src-model", type=str, default=None, help="Path to the Source Language SentencePiece tokenization model. If none, creates a tokenization model from the training-split.")
    parser.add_argument("--tgt-model", type=str, default=None, help="Path to the Target Language SentencePiece tokenization model. If none, creates a tokenization model from the training-split.")
    parser.add_argument("--force-train", action="store_true", help="Force training even if a model already exists.")
    # File prefixes (optional)
    parser.add_argument('--train-prefix', default=None, metavar='FP', help='raw train file prefix (without .lang extension)')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='raw tiny train file prefix (without .lang extension)')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='raw valid file prefix (without .lang extension)')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='raw test file prefix (without .lang extension)')
    parser.add_argument("--ignore-existing", action="store_true", help="Skip processing of raw-files if the output file already exists. Useful for resuming.")
    parser.add_argument("--src-vocab-size", type=int, default=32000, help="Vocabulary size for Source Language SentencePiece.")
    parser.add_argument("--tgt-vocab-size", type=int, default=32000, help="Vocabulary size for Target Language SentencePiece.")
    
    parser.add_argument("--quiet", action="store_true", help="Suppress logging output.")

    # OPTIONAL: add possibility to override default tokens
    parser.add_argument("--eos-token", type=str, default="</s>", help="End of sentence token.")
    parser.add_argument("--bos-token", type=str, default="<s>", help="Beginning of sentence token.")
    parser.add_argument("--pad-token", type=str, default="<pad>", help="Padding token.")
    parser.add_argument("--unk-token", type=str, default="<unk>", help="Unknown token.")
    
    return parser.parse_args()



def make_binary_dataset(input_file, output_file, preprocessor: BPETokenizer, append_eos=True, ignore_existing=False):
    # skip processing if output file already exists
    if os.path.exists(output_file) and not ignore_existing:
        logging.info(f"File {output_file} already exists, skipping...")
        return
    
    nsent, ntok = 0, 0
    # count unknown tokens
    unk_counter = 0
    
    # define consumer (used as a callback in encode_to_tensor) for unknown tokens
    def unk_consumer(idx):
        nonlocal unk_counter
        if idx == preprocessor.tokenizer.unk_id():
            unk_counter += 1

    tokens_list = []
    # open input file, read lines and encode_to_tensor
    with open(input_file, 'r', encoding='utf-8') as inf:
        for line in inf:
            tokens = preprocessor.encode_to_tensor(line.strip(), append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())
    # save output file
    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info(
    'Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by <unk>'.format(
        input_file, nsent, ntok, 0.0 if ntok == 0 else (100.0 * unk_counter / ntok)
    )
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # parse arguments from the command line
    args = get_args()

    # define paths for tokenization models
    # if no model path is given, create a model path in the model directory with the format LANG-bpe-VOCABSIZE.model
    tgt_tokenizer_model = args.tgt_model if args.tgt_model \
        else os.path.join(args.model_dir, f"{args.target_lang}-bpe-{args.tgt_vocab_size}.model")
    src_tokenizer_model = args.src_model if args.src_model \
        else os.path.join(args.model_dir, f"{args.source_lang}-bpe-{args.src_vocab_size}.model")
    
    os.makedirs(args.dest_dir, exist_ok=True)

    # ----------------------------
    # SOURCE LANGUAGE:
    src_processor = BPETokenizer(
        language=args.source_lang,
        vocab_size=args.src_vocab_size,
        eos=args.eos_token,
        bos=args.bos_token,
        pad=args.pad_token,
        unk=args.unk_token
    )
    # train or load model
    # if no model path is given or the given model file does not exist, train a new model
    if (not os.path.exists(src_tokenizer_model)) or (args.force_train):
        # error handling if no training data is provided
        if args.train_prefix is None:
            raise ValueError("No training data provided for training the source language tokenizer model.")
        # train model
        src_processor.train_tokenizer(training_data=os.path.join(args.raw_data, f"{args.train_prefix}.{args.source_lang}"), model_dir=args.model_dir)
        if not args.quiet:
            logging.info('Trained SentencePiece model for {} with {} words'.format(args.source_lang, src_processor.vocab_size))
    else:
        # load model
        src_processor.load(model_path=src_tokenizer_model)
        if not args.quiet:
            logging.info('Loaded SentencePiece model for {} from {}'.format(args.source_lang, src_tokenizer_model))
    src_processor.save_vocab(args.model_dir)
    # ----------------------------


    # ----------------------------
    # TARGET LANGUAGE:
    tgt_processor = BPETokenizer(
        language=args.target_lang,
        vocab_size=args.tgt_vocab_size,
        eos=args.eos_token,
        bos=args.bos_token,
        pad=args.pad_token,
        unk=args.unk_token
    )
    # train or load model
    if (not os.path.exists(tgt_tokenizer_model)) or (args.force_train):
        if args.train_prefix is None:
            raise ValueError("No training data provided for training the target language tokenizer model.")
        tgt_processor.train_tokenizer(training_data=os.path.join(args.raw_data, f"{args.train_prefix}.{args.target_lang}"), model_dir=args.model_dir)
        if not args.quiet:
            logging.info('Trained SentencePiece model for {} with {} words'.format(args.target_lang, tgt_processor.vocab_size))
    else:
        tgt_processor.load(model_path=tgt_tokenizer_model)
        if not args.quiet:
            logging.info('Loaded SentencePiece model for {} from {}'.format(args.target_lang, args.tgt_model))
    tgt_processor.save_vocab(args.model_dir)
    # ----------------------------

    # function to create dataset splits with the desired prefixes
    def make_split_datasets(lang, pre_processor):
        """create dataset splits (train, tiny_train, valid, test) for a given language using a dictionary"""
        if args.train_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, args.train_prefix + '.' + lang),
                output_file=os.path.join(args.dest_dir, 'train.' + lang),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing
                )
        if args.tiny_train_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, args.tiny_train_prefix + '.' + lang),
                output_file=os.path.join(args.dest_dir, 'tiny_train.' + lang),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing
                )
        if args.valid_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, args.valid_prefix + '.' + lang),
                output_file=os.path.join(args.dest_dir, 'valid.' + lang),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing
                )
        if args.test_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, args.test_prefix + '.' + lang),
                output_file=os.path.join(args.dest_dir, args.test_prefix + '.' + lang),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing
                )
    
    make_split_datasets(args.source_lang, src_processor)
    make_split_datasets(args.target_lang, tgt_processor)

    logging.info('Data processing complete!')
