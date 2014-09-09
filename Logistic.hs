{-# OPTIONS_GHC -w #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-type-defaults #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind #-}

{-# LANGUAGE TupleSections #-}

module Logistic where

import Numeric.AD

import qualified Data.Vector as V
import Control.Monad
import Control.Monad.State
import Data.List
import Text.Printf

import System.Random
import Data.Random ()
import Data.Random.Distribution.Beta
import Data.RVar

logit :: Floating a => a -> a
logit x = 1 / (1 + exp (negate x))

logLikelihood :: Floating a => 
                 V.Vector a -> 
                 a -> 
                 V.Vector a -> 
                 a
logLikelihood theta y x = y * log (logit z) + (1 - y) * log (1 - logit z)
	where
		z = V.sum $ V.zipWith (*) theta x

totalLogLikelihood :: Floating a =>
                      V.Vector a ->
                      V.Vector a ->
                      V.Vector (V.Vector a) ->
                      a
totalLogLikelihood theta y x = (a - delta * b)/l
	where
		l = fromIntegral $ V.length y
		a = V.sum $ V.zipWith (logLikelihood theta) y x
		b = (/2) $ V.sum $ V.map (^2) theta

delta :: Floating a => a
delta = 1.0

gamma :: Double
gamma = 0.1

nIters :: Int
nIters = 4000

delTotalLoglikelihood :: Floating a =>
                         V.Vector a ->
                         V.Vector (V.Vector a) ->
                         V.Vector a ->
                         V.Vector a
delTotalLoglikelihood y x = grad f
	where
		f theta = totalLogLikelihood theta 
		                             (V.map auto y) 
		                             (V.map (V.map auto) x)

stepOnce :: Double ->
            V.Vector Double ->
            V.Vector (V.Vector Double) ->
            V.Vector Double ->
            V.Vector Double
stepOnce gamma y x theta = 
	V.zipWith (+) theta (V.map (* gamma) $ del theta)
	where
		del = delTotalLoglikelihood y x

estimates :: (Floating a, Ord a) =>
             V.Vector a ->
             V.Vector (V.Vector a) ->
             V.Vector a ->
             [V.Vector a]
estimates y x = gradientAscent $ \theta -> totalLogLikelihood theta
                                                              (V.map auto y)
                                                              (V.map (V.map auto) x)

betas :: Int -> Double -> Double -> [Double]
betas n a b =
	evalState (replicateM n (sampleRVar (beta a b))) (mkStdGen seed)
	 where
	 	seed = 0

a, b :: Double
a = 15
b = 6
nSamples :: Int
nSamples = 100000

sample0, sample1 :: [Double]
sample0 = betas nSamples a b
sample1 = betas nSamples b a

mixSamples :: [Double] -> [Double] -> [(Double, Double)]
mixSamples xs ys = unfoldr g (map (0,) xs, map (1,) ys)
	where
		g ([], []) = Nothing
		g ([], _)  = Nothing
		g (_, [])  = Nothing
		g (x:xs, y) = Just (x, (y, xs))

createSample :: V.Vector (Double, Double)
createSample = V.fromList $ take 100 $ mixSamples sample1 sample0

actualTheta :: V.Vector Double
actualTheta = V.fromList [0.0, 1.0]

initTheta :: V.Vector Double
initTheta = V.replicate (V.length actualTheta) 0.1

vals :: V.Vector (Double, V.Vector Double)
vals = V.map (\(y, x) -> (y, V.fromList [1.0, x])) createSample

main :: IO ()
main = do
  let u = V.map fst vals
      v = V.map snd vals
      hs = iterate (stepOnce gamma u v) initTheta
      xs = V.map snd vals
      theta =  head $ drop nIters hs
      theta' = estimates u v initTheta !! 100
  printf "Hand crafted descent: theta_0 = %5.3f, theta_1 = %5.3f\n"
         (theta V.! 0) (theta V.! 1)
  printf "Library descent:      theta_0 = %5.3f, theta_1 = %5.3f\n"
         (theta' V.! 0) (theta' V.! 1)
  let predProbs  = V.map (logit . V.sum . V.zipWith (*) theta) xs
      mismatches = V.filter (> 0.5) $
                   V.map abs $
                   V.zipWith (-) actuals preds
        where
          actuals = V.map fst vals
          preds   = V.map (\x -> fromIntegral $ fromEnum (x > 0.5))
                          predProbs
  let lActuals, lMisMatches :: Double
      lActuals    = fromIntegral $ V.length vals
      lMisMatches = fromIntegral $ V.length mismatches
  printf "%5.2f%% correct\n" $
         100.0 *  (lActuals - lMisMatches) / lActuals