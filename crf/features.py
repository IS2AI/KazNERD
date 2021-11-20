def word2features(sent, i, use_context):
    word = sent[i][0]

    features = {
        #word shape
        'word.lower()': word.lower(),
        'word.istitle()': word.istitle(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'word.isalnum()': word.isalnum(),
        #suffixes
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-1:]': word[-1:],
        #prefixes
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[:1]': word[:1],
    }

    if use_context:
        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.isalpha()': word1.isalpha(),
                '-1:word.isalnum()': word1.isalnum(),
                })
        else:
            features['BOS'] = True

        if i > 1:
            word1 = sent[i-2][0]
            features.update({
                '-2:word.lower()': word1.lower(),
                '-2:word.istitle()': word1.istitle(),
                '-2:word.isupper()': word1.isupper(),
                '-2:word.isdigit()': word1.isdigit(),
                '-2:word.isalpha()': word1.isalpha(),
                '-2:word.isalnum()': word1.isalnum(),
            })
        else:
            features['BOS2'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.isalpha()': word1.isalpha(),
                '+1:word.isalnum()': word1.isalnum(),
            })
        else:
            features['EOS'] = True
   
        if i < len(sent)-2:
            word1 = sent[i+2][0]
            features.update({
                '+2:word.lower()': word1.lower(),
                '+2:word.istitle()': word1.istitle(),
                '+2:word.isupper()': word1.isupper(),
                '+2:word.isdigit()': word1.isdigit(),
                '+2:word.isalpha()': word1.isalpha(),
                '+2:word.isalnum()': word1.isalnum(),
            })
        else:
            features['EOS2'] = True

    return features
