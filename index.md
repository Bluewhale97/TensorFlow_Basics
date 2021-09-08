{
 "cells": [
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "                                      Introduction to tensorflow\n",
    "\n",
    "Using tensorflow to model a relationship between celsius and farenheit:\n",
    "\n",
    "Farenheit = 1.8 * Celsius + 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "## tensorflow importing\n",
    "\n",
    "import tensorflow as tf\n",
    "tf.logging.set_verbosity(tf.logging.ERROR)\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "## creting list of X and Y\n",
    "\n",
    "celsius_q=np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)\n",
    "fahrenheit_a=np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Creating a single layer\n",
    "\n",
    "l0 = tf.keras.layers.Dense(units=1, input_shape=[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Defining the order of training \n",
    "\n",
    "model = tf.keras.Sequential([l0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Defining the optimizing method \n",
    "\n",
    "model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "## model fitting\n",
    "\n",
    "history=model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-44.452038],\n",
       "       [ 10.350399],\n",
       "       [ 28.617878],\n",
       "       [ 43.23186 ],\n",
       "       [ 56.019096],\n",
       "       [ 68.806335],\n",
       "       [ 98.0343  ]], dtype=float32)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Predicting the training observation\n",
    "\n",
    "model.predict(celsius_q)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x263ae04f3c8>]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## visual display of loss vs epoch or iteration\n",
    "\n",
    "plt.xlabel('Epoch Number')\n",
    "plt.ylabel('Loss Magnitude')\n",
    "plt.plot(history.history['loss'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array([[1.8267479]], dtype=float32), array([28.617878], dtype=float32)]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Displaying the weights\n",
    "\n",
    "l0.get_weights()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.6 (tensorflow)",
   "language": "python",
   "name": "tensorflow"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
