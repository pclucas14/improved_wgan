from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['tensor_shape'] = (64, 3, 64, 64)
params['initial_eta'] = 1e-4
params['lambda'] = 10
params['load_weights'] = None#(9,200) #(12, 1000)#None# version/epoch tupple pair
params['optimizer'] = 'adam'
params['image_prepro'] = 'DCGAN' # (/250.; -0.5; /0.5) taken from DCGAN repo.
params['loss_comments'] = 'improved wgan with gradient penalty'
params['epoch_iter'] = 25
params['gen_iter'] = 1
params['critic_iter'] = 1
params['test'] = False

generator_layers = generator(batch_size=params['batch_size'])
critic_layers = critic(batch_size=params['batch_size'])
generator = generator_layers[-1]
critic = critic_layers[-1]

dh = DataHandler(tensor_shape=params['tensor_shape'])
eh = ExpHandler(params, test=params['test'])

# placeholders 
images = T.tensor4('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))
theano_rng = RandomStreams(rng.randint(2 ** 15))
alpha = theano_rng.uniform(size=(params['batch_size'],1), low=0., high=1.)

# params
gen_params = ll.get_all_params(generator, trainable=True)
critic_params = ll.get_all_params(critic, trainable=True)

# outputs
fake_images = ll.get_output(generator)
real_out = ll.get_output(critic, inputs=images)
fake_out = ll.get_output(critic, inputs=fake_images)
'''
gen_loss = -fake_out.mean()
critic_loss = fake_out.mean() - real_out.mean()

differences = fake_images - images
interpolates = images + (alpha * differences)
interpolates_out = ll.get_output(critic, inputs=interpolates)
gradients = theano.grad(interpolates_out.sum(), wrt=interpolates)

# TODO : check slopes shape
slopes = T.sqrt(T.sum((gradients ** 2)))
gradient_penalty = ((slopes-1.)**2).mean()
critic_loss += params['lambda'] * gradient_penalty
'''

gen_loss = lasagne.objectives.squared_error(fake_out, 1).mean()
critic_loss = (lasagne.objectives.squared_error(real_out, 1) + 
               lasagne.objectives.squared_error(fake_out, 0)).mean()



gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)

# updates
updates_gen = optimizer_factory(params['optimizer'], gen_grads, gen_params, eta)
updates_critic = optimizer_factory(params['optimizer'], critic_grads, critic_params, eta)

# function outputs
fn_output = OrderedDict()
fn_output['gen_loss'] = gen_loss
fn_output['critic_loss'] = critic_loss
# fn_output['gradient_norm'] = slopes
# fn_output['critic_grad'] = critic_grads_norm
fn_output['gen_grad'] = gen_grads_norm

eh.add_model('gen', generator_layers, fn_output)
eh.add_model('cri', critic_layers, fn_output)

# functions
print 'compiling functions'
train_gen = theano.function(inputs=[index], 
                              outputs=fn_output.values(), 
                              updates=updates_gen, 
                              givens={images: dh.GPU_image[index * params['batch_size']: 
                                                       (index+1) * params['batch_size']]},
                              name='train_gen')

train_critic = theano.function(inputs=[index], 
                              outputs=fn_output.values(), 
                              updates=updates_critic, 
                              givens={images: dh.GPU_image[index * params['batch_size']: 
                                                       (index+1) * params['batch_size']]},
                              name='train_cri')

test_gen =     theano.function(inputs=[],
                               outputs=[fake_images],
                               name='test_gen')

'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    gen_err, critic_err = 0, 0

    for _ in range(params['epoch_iter']):
        for _ in range(params['gen_iter']): 
            batch_no = dh.get_next_batch_no()
            model_out = train_gen(batch_no)

            # import pdb; pdb.set_trace()
            gen_err += np.array(model_out)
            eh.record('gen', np.array(model_out))

        for _ in range(params['critic_iter']):
            batch_no = dh.get_next_batch_no()
            model_out = train_critic(batch_no)

            critic_err += np.array(model_out)
            eh.record('cri', np.array(model_out))

    # test out model
    batch_no = dh.get_next_batch_no()
    eh.save_image(test_gen()[0])
    eh.end_of_epoch()
    
    print epoch
    print("gen  loss:\t\t{}".format(gen_err / (params['gen_iter']*params['epoch_iter'])))
    print("critic  loss:\t\t{}".format(critic_err / (params['critic_iter']*params['epoch_iter'])))
 
