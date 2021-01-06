FROM wav2letter/wav2letter:cpu-latest

ENV USE_CUDA=0
ENV KENLM_ROOT_DIR=/root/kenlm

# will use Intel MKL for featurization but this may cause dynamic loading conflicts.
# ENV USE_MKL=1

ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64:$LD_IBRARY_PATH
WORKDIR /root/wav2letter/bindings/python

RUN pip install --upgrade pip && pip install soundfile packaging && pip install -e .

WORKDIR /root
RUN git clone https://github.com/pytorch/fairseq.git
RUN mkdir data
COPY examples/wav2vec/recognize.py /root/fairseq/examples/wav2vec/recognize.py

WORKDIR /root/fairseq
RUN pip install --editable ./ && python examples/speech_recognition/infer.py --help && python examples/wav2vec/recognize.py --help
