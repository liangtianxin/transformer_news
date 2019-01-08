arg.is_training = False
arg.dropout_rate = 0.

g = Graph(arg)

saver =tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'logs/model')
    while True:
        line = input('input: ')
        if line == 'exit': break
        line = line.lower().replace(',', ' ,').strip('\n').split(' ')
        x = np.array([en_vocab.index(pny) for pny in line])
        x = x.reshape(1, -1)
        de_inp = [[zh_vocab.index('<GO>')]]
        while True:
            y = np.array(de_inp)
            preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
            if preds[0][-1] == zh_vocab.index('<EOS>'):
                break
            de_inp[0].append(preds[0][-1])
        got = ''.join(zh_vocab[idx] for idx in de_inp[0][1:])
        print(got)
