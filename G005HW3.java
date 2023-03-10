import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.util.random.PoissonSampler;

import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class G005HW3 {

    private static double[][] distances; // Pre-computed distances

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // MAIN PROGRAM
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Print input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end - start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end - start) + " ms");

    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // AUXILIARY METHODS
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method strToVector: input reading
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method euclidean: distance function
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method MR_kCenterOutliers: MR algorithm for k-center with outliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers(JavaRDD<Vector> points, int k, int z, int L) {

        // ------------- ROUND 1 ---------------------------
        // Start of time measurement for round 1
        Long start_round_1 = System.currentTimeMillis();

        JavaRDD<Tuple2<Vector, Long>> coreset = points.mapPartitions(x -> {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext())
                partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k + z + 1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }).cache();

        // ------------- END OF ROUND 1 ---------------------------

        // ------------- ROUND 2 ---------------------------
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z) * L);
        elems.addAll(coreset.collect());

        // End of time measurement for round 1 and start of time measurement for round
        // 2.
        // The time measurement for round 1 is stopped here due to lazy evaluation.
        long end_round_1 = System.currentTimeMillis();

        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        ArrayList<Vector> elems_points = new ArrayList<>();
        ArrayList<Long> elems_weights = new ArrayList<>();
        for (int i = 0; i < elems.size(); i++) {
            elems_points.add(elems.get(i)._1());
            elems_weights.add(elems.get(i)._2());
        }

        distances = new double[elems_points.size()][elems_points.size()];
        PrecomputeDistances(elems_points);
        ArrayList<Vector> final_centers = SeqWeightedOutliers(elems_points, elems_weights, k, z, 2);

        // End of time measurement for round 2
        long end_round_2 = System.currentTimeMillis();

        // ------------- END OF ROUND 2 ---------------------------
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        System.out.println("Time Round 1: " + (end_round_1 - start_round_1) + " ms");
        System.out.println("Time Round 2: " + (end_round_2 - end_round_1) + " ms");

        // ****** Return the final solution
        return final_centers;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method kCenterFFT: Farthest-First Traversal
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT(ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius = 0;

        for (int iter = 1; iter < k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method computeWeights: compute weights of coreset points
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " +
            // centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method SeqWeightedOutliers: sequential k-center with outliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z,
                                                        double alpha) {
        // Computation print of r_min
        double min_distance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < k + z + 1; i++)
            for (int j = i + 1; j < k + z + 1; j++)
                if (distances[i][j] < min_distance)
                    min_distance = distances[i][j];
        double r = min_distance / 2;
        System.out.println("Initial guess = " + r); // initial guess r
        // Initialization of the number of guesses for r
        int guessCount = 1;
        // Evaluation of S
        while (true) {
            // Initialization of Z, S and W_z
            ArrayList<Vector> Z = new ArrayList<>(P); // Z <- P
            ArrayList<Vector> S = new ArrayList<>(); // S <- 0
            double W_z = 0;
            for (Long w : W)
                W_z += w;
            // Evaluation of S with the current r
            while (S.size() < k && W_z > 0) {
                double max = 0;
                int newCenterIndex = -1;
                // Seeking the first center
                for (int i = 0; i < P.size(); i++) {
                    double ballWeight = 0;
                    // Computation of the ball-weight
                    for (int j = 0; j < Z.size(); j++)
                        // Checking that Z.get(j) has not been removed from Z yet
                        if (Z.get(j) != null)
                            if (distances[i][j] <= (1 + 2 * alpha) * r)
                                ballWeight += W.get(j);
                    // Update of max if necessary
                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenterIndex = i;
                    }
                }
                // Insert newCenter in S and remove points outside the ball from Z
                S.add(P.get(newCenterIndex));
                for (int i = 0; i < Z.size(); i++) {
                    if (Z.get(i) != null) {
                        if (distances[newCenterIndex][i] <= (3 + 4 * alpha) * r) {
                            // Removal of Z.get(i) by setting it to null
                            Z.set(i, null);
                            W_z -= W.get(i);
                        }
                    }
                }
            }
            // Check if S is the final solution
            if (W_z <= z) {
                System.out.println("Final guess = " + r); // final guess r
                System.out.println("Number of guesses = " + guessCount); // SeqWeightedOutliers guess
                return S;
            } else {
                r = 2 * r;
                guessCount++;
            }
        }
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Distance Pre-computation
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    /*
     * The distances among all the points in P are computed and put in an upper
     * triangular matrix.
     * Since distance(x, y) = distance(y, x) for all x, y in P, the lower triangular
     * matrix is initialized symmetrically.
     */
    public static void PrecomputeDistances(ArrayList<Vector> P) {
        for (int i = 0; i < P.size(); i++) {
            for (int j = i; j < P.size(); j++) {
                double distance = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
                distances[i][j] = distance;
                // Distance is commutative
                distances[j][i] = distance;
            }
        }
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method computeObjective: computes objective function
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective(JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {
        JavaDoubleRDD objective = points.mapToDouble(x -> {
            Double min_distance = Double.MAX_VALUE;
            for (Vector center : centers) {
                Double current_distance = euclidean(x, center);
                if (current_distance < min_distance) {
                    min_distance = current_distance;
                }
            }
            return min_distance;
        });

        return objective.top(z + 1).get(z);
    }

    //ALTERNATIVE METHOD
    //This method considers (key, value) pairs instead of only values.
    //We decided to use the above method instead of this one because it is more efficient both in term of space and time.
    //Indeed The below method does waste time and space because it considers unused integers (all set to 0) in the pairs.
    //We wanted to add also this function because theoretically speaking is more similar to what we have studied during the course.
    /*
    public static double computeObjective(JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {
        List<Tuple2<Double, Integer>> distances = points.mapToPair(point -> {
            double current_key = Double.POSITIVE_INFINITY;
            for (Vector center : centers) {
                double current_distance = euclidean(point, center);
                if (current_distance < current_key)
                    current_key = current_distance;
            }
            return new Tuple2<>(current_key, 0);
        }).sortByKey(false).take(z + 1);
        return distances.get(z)._1();
    }
    */
}
