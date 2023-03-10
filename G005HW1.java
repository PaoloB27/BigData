import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import shapeless.Tuple;

import java.io.IOException;
import java.util.*;

public class G005HW1 {

    public static void main(String[] args) throws IOException {

        // Checking number of cmd line parameters
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions num_top country file_path");
        }

        // Spark conf & context setup
        SparkConf conf = new SparkConf(true).setAppName("G005HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        int H = Integer.parseInt(args[1]);
        String S = args[2];

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 1
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache(); // Create RDD

        // Print the number of rows read from the input file
        System.out.println("Number of rows = " + rawData.count());

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 2
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Extract key-value pairs from each transition
        JavaPairRDD<String, Integer> productCustomer = rawData
                .flatMapToPair((transaction) -> {
                    String tokens[] = transaction.split(",");

                    ArrayList<Tuple2<Tuple2<String, Integer>, Integer>> pairs = new ArrayList<>();

                    String P = tokens[1];
                    int quantity = Integer.parseInt(tokens[3]);
                    Integer C = Integer.parseInt(tokens[6]);

                    if ((quantity > 0) && (S.equalsIgnoreCase("all") || S.equalsIgnoreCase(tokens[7]))) {
                        pairs.add(new Tuple2<>(new Tuple2<>(P, C), 0));
                    }

                    return pairs.iterator();
                })
                .groupByKey()
                .mapToPair((pairKey) -> {
                    Tuple2<String, Integer> pair = new Tuple2<String, Integer>(pairKey._1()._1(), pairKey._1()._2());
                    return pair;
                })
                .cache();

        // Print the number of product-costumer pairs
        System.out.println("Product-Customer Pairs = " + productCustomer.count());

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 3
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        JavaPairRDD<String, Integer> productPopularity1;

        productPopularity1 = productCustomer
                .mapPartitionsToPair((element) -> {
                    HashMap<String, Integer> popularityMap = new HashMap<>();
                    while (element.hasNext()) {
                        Tuple2<String, Integer> product = element.next();
                        String productID = product._1();
                        if (popularityMap.containsKey(productID)) {
                            popularityMap.put(productID, popularityMap.get(productID) + 1);
                        } else {
                            popularityMap.put(productID, 1);
                        }
                    }
                    ArrayList<Tuple2<String, Integer>> popularityList = new ArrayList<>();
                    for (Map.Entry<String, Integer> i : popularityMap.entrySet()) {
                        popularityList.add(new Tuple2<String, Integer>(i.getKey(), i.getValue()));
                    }
                    return popularityList.iterator();
                })
                .groupByKey()
                .mapValues((it) -> {
                    Integer popularity = 0;
                    for (Integer i : it) {
                        popularity += i;
                    }
                    return popularity;
                });

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 4
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Integer> productPopularity2;

        productPopularity2 = productCustomer
                .mapToPair((pair) -> {
                    Tuple2<String, Integer> newPair = new Tuple2<>(pair._1(), 1);
                    return newPair;
                })
                .reduceByKey((x, y) -> x + y);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 5
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (H > 0) {

            // Reverse key and value and then order the RDD by key. Finally, extract the H
            // highest keys.
            // ArrayList<Tuple2<Integer, String>> ciao = productPopularity1.mapToPair((pair)
            // -> {
            // return new Tuple<>(pair._2(), pair._1());
            // });

            List<Tuple2<Integer, String>> HpopularPairsRev = productPopularity1
                    .mapToPair((pair) -> {
                        Tuple2<Integer, String> newPair = new Tuple2<>(pair._2(), pair._1());
                        return newPair;
                    })
                    .sortByKey(false)
                    .take(H);

            // Reverse again the pairs' elements to get the final right order
            ArrayList<Tuple2<String, Integer>> HpopularPairs = new ArrayList<>();
            for (Tuple2<Integer, String> pair : HpopularPairsRev) {
                HpopularPairs.add(new Tuple2<>(pair._2(), pair._1()));
            }
            // Print the top-H products with their ID and popularity
            System.out.println("Top 5 Products and their Popularities");
            for (Tuple2<String, Integer> pair : HpopularPairs) {
                System.out.print("Product " + pair._1() + " Popularity " + pair._2() + "; ");
            }
            System.out.println();
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 6
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Collect pairs from productPopularity1 in a list, then prints them in
        // lexicographic order
        // Then does teh same for productPopularity2
        if (H == 0) {
            TreeMap<String, Integer> sortedProductPopularity = new TreeMap<>();

            // Processes productPopularity1 data
            sortedProductPopularity.putAll(productPopularity1.collectAsMap());

            System.out.println("productPopularity1: ");

            for (Map.Entry<String, Integer> entry : sortedProductPopularity.entrySet()) {
                System.out.print("Product: " + entry.getKey() + " Popularity: " + entry.getValue() + "; ");
            }

            System.out.println();

            sortedProductPopularity.clear(); // clears data in map

            // Processes productPopularity2 data
            sortedProductPopularity.putAll(productPopularity2.collectAsMap());

            System.out.println("productPopularity2: ");

            for (Map.Entry<String, Integer> entry : sortedProductPopularity.entrySet()) {
                System.out.print("Product: " + entry.getKey() + " Popularity: " + entry.getValue() + "; ");
            }

            System.out.println();
        }
    }
}