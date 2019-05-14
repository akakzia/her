import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import time


def visualize_evaluations(table, save=False):
    if save:
        print("Saving")
    x = np.arange(0, len(table), 1)
    plt.plot(x, table, '-r')
    plt.ylim(0, 1)
    plt.title('Success rate per epoch')
    plt.grid()
    plt.show()


# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    r = tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    q = tf.summary.scalar("Qmax Value", episode_ave_max_q)
    success_rate = tf.Variable(0.)
    s = tf.summary.scalar("Success rate", success_rate)

    summary_vars_1 = [episode_reward, episode_ave_max_q]
    summary_vars_2 = [success_rate]
    summary_first = tf.summary.merge([r, q])
    summary_second = tf.summary.merge([s])

    return summary_first, summary_second, summary_vars_1, summary_vars_2


# ===========================
#   Agent Training
# ===========================

def train(sess, saver, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_first, summary_second, summary_vars_1, summary_vars_2 = build_summaries()

    sess.run(tf.global_variables_initializer())
    if args['HER']:
        args['summary_dir'] = args['summary_dir'] + '+her'
    writer = tf.summary.FileWriter("{0}-{1}".format(args['summary_dir'], int(time.time())), sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    # evaluations = [0]
    for epoch in range(int(args['epochs'])):
        print('=========== EPOCH {:d} ==========='.format(epoch+1))
        success = 0.
        tasks = env.unwrapped.sample_tasks(int(args['max_episodes']))
        for i in range(int(args['max_episodes'])):

            s = env.reset_task(tasks[i])
            ep_reward = 0
            ep_ave_max_q = 0
            episode = []
            for j in range(int(args['max_episode_len'])):

                if args['render_env']:
                    env.render()

                # Added exploration noise
                # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
                k = np.random.uniform(0, 1)
                if k < 0.1:
                    a = [np.random.uniform(-1, 1, 2)]
                else:
                    a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
                s2, r, terminal, info = env.step(a[0])
                episode.append((s, r, terminal, s2))
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > int(args['minibatch_size']):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(int(args['minibatch_size']))

                    # Calculate targets
                    target_q = critic.predict_target(
                        s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(int(args['minibatch_size'])):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + critic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(
                        s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                s = s2
                ep_reward += r

                if terminal:
                    if ep_reward == 1:
                        success += 1
                    summary_str_1 = sess.run(summary_first, feed_dict={
                        summary_vars_1[0]: ep_reward,
                        summary_vars_1[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str_1, i + int(args['max_episodes']) * epoch)
                    writer.flush()

                    print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                          i, (ep_ave_max_q / float(j))))
                    if args['HER']:
                        """for state, reward, done, next_state in episode:
                            new_goal = next_state
                            fictive_reward = 1
                            d = True
                            new_state = np.concatenate((state[:4], new_goal[:4]))
                            new_next_state = np.concatenate((next_state[:4], new_goal[:4]))
                            replay_buffer.add(np.reshape(new_state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), fictive_reward,
                                              d, np.reshape(new_next_state, (actor.s_dim,)))"""
                        if len(episode) == int(args['max_episode_len']):
                            for t in np.random.choice(len(episode), 10):
                                new_goal = episode[t][-1]
                                for state, reward, done, next_state in episode[:t]:
                                    new_state = np.concatenate((state[:4], new_goal[:4]))
                                    new_next_state = np.concatenate((next_state[:4], new_goal[:4]))
                                    fictive_reward = 0
                                    d = False
                                    if new_next_state.all() == new_goal.all():
                                        fictive_reward = 1
                                        d = True
                                    replay_buffer.add(np.reshape(new_state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)),
                                                      fictive_reward, d, np.reshape(new_next_state, (actor.s_dim,)))
                    break
        success_rate = success / int(args['max_episodes'])
        summary_str_2 = sess.run(summary_second, feed_dict={
            summary_vars_2[0]: success_rate
        })

        writer.add_summary(summary_str_2, epoch)
        writer.flush()
        if epoch % 10 == 0:
            save_path = saver.save(sess, './models/modified/model-{0}.ckpt'.format(epoch))
            print("Model saved in path: %s" % save_path)
        # evaluations.append(success_rate)
    # visualize_evaluations(evaluations, save=False)
