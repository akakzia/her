import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


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
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    evaluations = [0]
    for epoch in range(int(args['epochs'])):
        print('=========== EPOCH {:d} ==========='.format(epoch+1))
        success = 0.
        for i in range(int(args['max_episodes'])):

            s = env.reset()
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
                    a = np.random.uniform(-1, 1, 2)
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
                    if ep_reward > -int(args['max_episode_len']):
                        success += 1
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                            i, (ep_ave_max_q / float(j))))
                    for state, reward, done, next_state in episode:
                        new_goal = next_state
                        fictive_reward = 0
                        d = True
                        new_state = np.concatenate((state[:4], new_goal[:4]))
                        new_next_state = np.concatenate((next_state[:4], new_goal[:4]))
                        replay_buffer.add(np.reshape(new_state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), fictive_reward,
                                          d, np.reshape(new_next_state, (actor.s_dim,)))
                    break
        success_rate = success / int(args['max_episodes'])
        evaluations.append(success_rate)
    visualize_evaluations(evaluations, save=False)
