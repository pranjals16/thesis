#this function will convert text to lowercase and will disconnect punctuation and special symbols from words
cd liblinear-1.94
make
cd ..
./liblinear-1.94/train -s 0 -c 8.5 train_average.txt model.logreg
./liblinear-1.94/predict -b 1 test_average.txt model.logreg out.logreg
tail -n 25000 out.logreg > SENTENCE-VECTOR.LOGREG

#cd ..
cat RNNLM-SCORE | awk ' \
BEGIN{cn=0; corr=0;} \
{ \
  if ($3<1) if (cn<12500) corr++; \
  if ($3>1) if (cn>=12500) corr++; \
  cn++; \
} \
END{print "RNNLM accuracy: " corr/cn*100 "%";}'

cat SENTENCE-VECTOR.LOGREG | awk ' \
BEGIN{cn=0; corr=0;} \
{ \
  if ($2>0.5) if (cn<12500) corr++; \
  if ($2<0.5) if (cn>=12500) corr++; \
  cn++; \
} \
END{print "Sentence vector + logistic regression accuracy: " corr/cn*100 "%";}'

paste RNNLM-SCORE SENTENCE-VECTOR.LOGREG | awk ' \
BEGIN{cn=0; corr=0;} \
{ \
  if (($3-1)*7+(0.5-$5)<0) if (cn<12500) corr++; \
  if (($3-1)*7+(0.5-$5)>0) if (cn>=12500) corr++; \
  cn++; \
} \
END{print "FINAL accuracy: " corr/cn*100 "%";}'
