import os
import tensorflow as tf
import time
import numpy as np
import math
import utils
import config
import evaluation
from generator import Generator
from discriminator import Discriminator
import warnings


from differential_privacy.privacy_accountant.tf import accountant
from differential_privacy.optimizer import our_dp_optimizer_MomentAcc
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
warnings.filterwarnings('ignore')
_,_,_,_,egs=utils.read_graph(config.train_file)
accountant = accountant.GaussianMomentsAccountant(config.n_node)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
class Model():
    def __init__(self,epsilon,sigma,clipping):
        t = time.time()
        print('reading graph...')
        self.graph, self.n_node, self.node_list, self.node_list_s, self.egs = utils.read_graph(config.train_file)
        self.node_emd_shape = [2, self.n_node, config.n_emb]
        print('[%.2f] reading graph finished. #node = %d' % (time.time() - t, self.n_node))
        self.epsilon=epsilon
        self.sigma=sigma

        self.clipping=clipping
        self.dis_node_embed_init = None
        self.gen_node_embed_init = None
        if config.pretrain_dis_node_emb:
            t = time.time()
            print('reading initial embeddings...')
            dis_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb) \
                                            for x in [config.pretrain_dis_node_emb]])
            gen_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb) \
                                            for x in [config.pretrain_gen_node_emb]])
            print('[%.2f] read initial embeddings finished.' % (time.time() - t))

        print('building DGGAN model...')
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()
        if config.experiment == 'link_prediction':
            self.link_prediction = evaluation.LinkPrediction(config)


        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=self.config)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)
        if config.pretrain_ckpt:
            print('restore...')
            pretrain_ckpt = tf.train.latest_checkpoint(config.pretrain_ckpt)
            self.saver.restore(self.sess, pretrain_ckpt)
        else:
            print('initial...')
            self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                    tf.compat.v1.local_variables_initializer())
            self.sess.run(self.init_op)

    def build_discriminator(self):
        # with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node=self.n_node,
                                           node_emd_init=self.dis_node_embed_init,
                                           config=config,clipping=self.clipping,sigma=self.sigma)

    def build_generator(self):
        # with tf.variable_scope("generator"):
        self.generator = Generator(n_node=self.n_node,
                                   node_emd_init=self.gen_node_embed_init,
                                   config=config)

    def train_dis(self, dis_loss, pos_loss, neg_loss, dis_cnt,d_epoch):
        np.random.shuffle(self.egs)
        should_terminate=False

        info = ''
        X=math.floor(len(self.egs) / config.d_batch_size)
        index=0
        while(index<X and should_terminate==False):
            #for index in range(math.floor(len(self.egs) / config.d_batch_size)):
            #if(not should_terminate):
                pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = self.prepare_data_for_d(index, self.egs)
                _, _loss, _pos_loss, _neg_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, \
                                                            self.discriminator.pos_loss, self.discriminator.neg_loss],
                                                           feed_dict={
                                                               self.discriminator.pos_node_ids: np.array(pos_node_ids),
                                                               self.discriminator.pos_node_neighbor_ids: np.array(
                                                                   pos_node_neighbor_ids),
                                                               self.discriminator.fake_node_embedding: np.array(
                                                                   fake_node_embedding)})
                # Flag to terminate based on target privacy budget
                terminate_spent_eps_delta = accountant.get_privacy_spent(self.sess,target_eps=[(self.epsilon)])
                terminate_spent_eps_delta=terminate_spent_eps_delta[0]
                # For the Moments accountant, we should always have spent_eps == max_target_eps.
                if (terminate_spent_eps_delta.spent_delta > config.delta or terminate_spent_eps_delta.spent_eps > self.epsilon):
                    should_terminate = True
                    print("epoch : " ,d_epoch)
                    print("TERMINATE!!! Run out of privacy budget ...")

                    spent_eps_deltas = accountant.get_privacy_spent(self.sess, target_eps=[(self.epsilon)])
                    print("Spent Eps and Delta : " + str(spent_eps_deltas))
                    d_epoch = config.d_epoch
                    break

                dis_loss += _loss
                pos_loss += _pos_loss
                for i in range(4):
                        neg_loss[i] += _neg_loss[i]
                dis_cnt += 1
                info = 'dis_loss=%.4f pos_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f neg_loss_2=%.4f neg_loss_3=%.4f' % \
                   (dis_loss / dis_cnt, pos_loss / dis_cnt, neg_loss[0] / dis_cnt, neg_loss[1] / dis_cnt, \
                    neg_loss[2] / dis_cnt, neg_loss[3] / dis_cnt)
                self.my_print(info, True, 1)
                spent_eps_deltas = accountant.get_privacy_spent(self.sess, target_eps=[(self.epsilon)])
                print("Spent Eps and Delta : " + str(spent_eps_deltas))
                index=index+1
        return (dis_loss, pos_loss, neg_loss, dis_cnt,d_epoch)

    def train_gen(self, gen_loss, neg_loss, gen_cnt):
        np.random.shuffle(self.node_list)


        info = ''
        for index in range(math.floor(len(self.node_list) / config.g_batch_size)):
            node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(index, self.node_list)
            _, _loss, _neg_loss = self.sess.run(
                [self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                feed_dict={self.generator.node_ids: np.array(node_ids),
                           self.generator.noise_embedding: np.array(noise_embedding),
                           self.generator.dis_node_embedding: np.array(dis_node_embedding)})

            gen_loss += _loss
            for i in range(2):
                neg_loss[i] += _neg_loss[i]
            gen_cnt += 1
            info = 'gen_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f' % (
            gen_loss / gen_cnt, neg_loss[0] / gen_cnt, neg_loss[1] / gen_cnt)
            self.my_print(info, True, 1)
        return (gen_loss, neg_loss, gen_cnt)

    def train(self):
        best_auc = [[0] * 3, [0] * 3, [0] * 3]
        best_epoch = [-1, -1, -1]
        #tf.compat.v1.global_variables_initializer().run()
        should_terminate = False
        np.random.shuffle(self.egs)
        lots_per_epoch=len(self.egs)/config.lot_size
        step=0
        print('start traning...')
        while(step<config.n_epoch):
            epoch=step/lots_per_epoch
            info = 'epoch %d' % step
            self.my_print(info, False, 1)
            dis_loss = 0.0
            dis_pos_loss = 0.0
            dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
            dis_cnt = 0

            gen_loss = 0.0
            gen_neg_loss = [0.0, 0.0]
            gen_cnt = 0
            for d in range(config.d_epoch):
              ind=0
              while(ind <(math.floor(len(self.egs)/config.d_batch_size)) and should_terminate==False):
                print("batch-",ind)

                np.random.shuffle(self.egs)

                info = ''
                for index in range(math.floor(len(self.egs) / config.d_batch_size)):
                    pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = self.prepare_data_for_d(index, self.egs)
                    _, _loss, _pos_loss, _neg_loss = self.sess.run(
                        [self.discriminator.d_updates, self.discriminator.loss, \
                         self.discriminator.pos_loss, self.discriminator.neg_loss],
                        feed_dict={self.discriminator.pos_node_ids: np.array(pos_node_ids),
                                   self.discriminator.pos_node_neighbor_ids: np.array(pos_node_neighbor_ids),
                                   self.discriminator.fake_node_embedding: np.array(fake_node_embedding)})

                    dis_loss += _loss
                    dis_pos_loss += _pos_loss
                    for i in range(4):
                        dis_neg_loss[i] += _neg_loss[i]
                    dis_cnt += 1
                    info = 'dis_loss=%.4f pos_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f neg_loss_2=%.4f neg_loss_3=%.4f' % \
                           (dis_loss / dis_cnt, dis_pos_loss / dis_cnt, dis_neg_loss[0] / dis_cnt, dis_neg_loss[1] / dis_cnt, \
                            dis_neg_loss[2] / dis_cnt, dis_neg_loss[3] / dis_cnt)
                    self.my_print(info, True, 1)
                ind=ind+1
              self.my_print('', False, 1)
              auc = self.evaluate()


            # G-step
            for g_epoch in range(config.g_epoch):
              ind = 0
              np.random.shuffle(self.node_list)
              while (ind < (math.floor(len(self.node_list) / config.g_batch_size)) and should_terminate == False):
                print("batch-", ind)
            #for i in range(math.floor(len(self.egs) / config.d_batch_size)):
                node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(ind, self.node_list)
                _, _loss, _neg_loss = self.sess.run(
                    [self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                    feed_dict={self.generator.node_ids: np.array(node_ids),
                               self.generator.noise_embedding: np.array(noise_embedding),
                               self.generator.dis_node_embedding: np.array(dis_node_embedding)})

                gen_loss += _loss
                for i in range(2):
                    gen_neg_loss[i] += _neg_loss[i]
                gen_cnt += 1
                info = 'gen_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f' % (
                gen_loss / gen_cnt, gen_neg_loss[0] / gen_cnt, gen_neg_loss[1] / gen_cnt)
                self.my_print(info, True, 1)
                ind = ind + 1
              self.my_print('', False, 1)
            terminate_spend_eps_delta=accountant.get_privacy_spent(self.sess)[0]
            if(terminate_spend_eps_delta.spent_delta>config.delta or terminate_spend_eps_delta.spent_eps>max(config.target_eps)):
                    spend_eps_delta=accountant.get_privacy_spent(self.sess)
                    print("TERMINATE!!! OUT OF BUDGET")
                    print("termination step=",str(step))
                    should_terminate=True
                    break

            #info = 'dis_loss=%.4f dis_pos_loss=%.4f dis_neg_loss_0=%.4f dis_neg_loss_1=%.4f dis_neg_loss_2=%.4f dis_neg_loss_3=%.4f' % (dis_loss / dis_cnt, dis_pos_loss / dis_cnt, dis_neg_loss[0] / dis_cnt, dis_neg_loss[1] / dis_cnt,dis_neg_loss[2] / dis_cnt, dis_neg_loss[3] / dis_cnt)
            #self.my_print(info, True, 1)
            #info = 'gen_loss=%.4f gen_neg_loss_0=%.4f gen_neg_loss_1=%.4f' % (gen_loss / gen_cnt, gen_neg_loss[0] / gen_cnt, gen_neg_loss[1] / gen_cnt)
            #self.my_print(info, True, 1)
            print()
            #auc = self.evaluate()
            print("\n ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("epoch=",step)
            print("epsilon=",terminate_spend_eps_delta.spent_eps)
            print("delta=", terminate_spend_eps_delta.spent_delta)
            step=step+1
            if config.save:
                self.write_embeddings_to_file(step)
        print('training finished.')

    def prepare_data_for_d(self, index, egs):
        pos_node_ids = []
        pos_node_neighbor_ids = []

        for eg in egs[index * config.d_batch_size: (index + 1) * config.d_batch_size]:
            node_id, node_neighbor_id = eg

            pos_node_ids.append(node_id)
            pos_node_neighbor_ids.append(node_neighbor_id)

        # generate fake node
        fake_node_embedding = []

        noise_embedding = np.random.normal(0.0, self.sigma, (2, len(pos_node_ids), config.n_emb))
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                                 feed_dict={self.generator.node_ids: np.array(pos_node_ids),
                                                            self.generator.noise_embedding: np.array(noise_embedding)}))

        noise_embedding = np.random.normal(0.0, self.sigma, (2, len(pos_node_ids), config.n_emb))
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                                 feed_dict={self.generator.node_ids: np.array(pos_node_neighbor_ids),
                                                            self.generator.noise_embedding: np.array(noise_embedding)}))

        return pos_node_ids, pos_node_neighbor_ids, fake_node_embedding


    def prepare_data_for_g(self, index, node_list):
        node_ids = []

        for node_id in node_list[index * config.g_batch_size: (index + 1) * config.g_batch_size]:
            node_ids.append(node_id)

        noise_embedding = np.random.normal(0.0, self.sigma, (2, len(node_ids), config.n_emb))

        dis_node_embedding = []
        dis_node_embedding1 = self.sess.run([self.discriminator.pos_node_embedding],
                                            feed_dict={self.discriminator.pos_node_ids: np.array(node_ids)})
        dis_node_embedding2 = self.sess.run([self.discriminator.pos_node_neighbor_embedding],
                                            feed_dict={self.discriminator.pos_node_neighbor_ids: np.array(node_ids)})
        dis_node_embedding = np.vstack([dis_node_embedding1, dis_node_embedding2])
        return node_ids, noise_embedding, dis_node_embedding


    def evaluate(self):
        if config.experiment == 'link_prediction':
            return self.evaluate_link_prediction()

    def evaluate_link_prediction(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        auc = self.link_prediction.evaluate(embedding_matrix)
        info = 'auc_0=%.4f auc_50=%.4f auc_100=%.4f' % (auc[0], auc[1], auc[2])
        self.my_print(info, False, 1)
        self.link_prediction.inference_via_confidence(self.discriminator.node_embedding_matrix)
        return auc

    def write_embeddings_to_file(self, epoch):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        models = [self.generator, self.discriminator]
        emb_filenames = ['gen.txt', 'dis_s.txt', 'dis_t.txt']
        embedding_matrix = [self.sess.run(self.generator.node_embedding_matrix)]
        embedding_matrix.extend([self.sess.run(self.discriminator.node_embedding_matrix)[0],
                                 self.sess.run(self.discriminator.node_embedding_matrix)[1]])
        for i in range(3):
            index = np.array(range(self.n_node)).reshape(-1, 1)
            t = np.hstack([index, embedding_matrix[i]])
            embedding_list = t.tolist()
            embedding_str = [str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in
                             embedding_list]

            file_path = '%s%d-%s' % (config.save_path, epoch, emb_filenames[i])
            with open(file_path, 'w') as f:
                lines = [str(self.n_node) + ' ' + str(config.n_emb) + '\n'] + embedding_str
                f.writelines(lines)
        self.saver.save(self.sess, config.save_path + 'model.ckpt', global_step=epoch)

    def my_print(self, info, r_flag, verbose):
        if verbose == 1 and config.verbose == 0:
            return
        if r_flag:
            print('\r%s' % info, end='')
        else:
            print('%s' % info)


if __name__ == '__main__':
    #epsilons = [0.1,0.2,0.5,1.0,2.0,5.0,8.0,10.0]
    #sigmas = [0.1,0.3,0.5,0.7,0.9,1.0]
    #clippings = [1.0,2.0,3.0,4.0,5.0]
    epsilons=[1]
    sigmas=[0.5]
    clippings=[0.5]
    for epsilon in epsilons:
        for sigma in sigmas:
            for clipping in clippings:
                model = Model(epsilon,sigma,clipping)
                model.train()
            print("=========================================================================================================")
        print("*****************************************************************************************************************")
    print("##########################################################################################")
