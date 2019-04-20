package app;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class SimpleApp {
	public static void main(String[] args) throws IOException {

		// This file contains the data from
		// http://archive.ics.uci.edu/ml/datasets/Air+Quality
		// Place the file in root folder.
		String dataFile = "AirQualityUCI.csv";

		// Cleaned data will be stored in AirQualityUCI_WithLabels.csv
		String clenedDataFile = "AirQualityUCI_WithLabels.csv";

		// Data cleaning and preparation
		// Creating of an array of labels (column "T")
		String[] labels = new String[10000];

		int ind = 0;

		BufferedReader br_l = new BufferedReader(new FileReader(dataFile));
		String row = "";
		// int rowcount = 0;

		while ((row = br_l.readLine()) != null) {
			// rowcount++;
			String[] cols = row.split(";"); // Splitting by ;
			if (cols.length == 0)
				break; // Skipping the empty line data
			// We don't need the first line where it gives the headings about
			// the data columns.
			if (cols[0].contains("Date")) {
				continue;
			}
			;

			// Column #12 has all the temperatures of all the dates and times
			labels[ind] = cols[12].replaceAll(",", ".");
			ind = ind + 1;
		}
		// System.out.println("rowcount: "+rowcount);

		// for (int j = 0; j < labels.length; j++) {
		// if (j == 0) System.out.println("$$$ "+ labels[j]);
		// }

		// appending the labels column to the original data table (for every row
		// we append the label that equal to the temp on the next day at the
		// same time)

		BufferedReader br = new BufferedReader(new FileReader(dataFile));

		BufferedWriter bw = null;
		FileWriter fw = null;

		fw = new FileWriter(clenedDataFile);
		bw = new BufferedWriter(fw);

		String l = "";
		int ind2 = 24;

		while ((l = br.readLine()) != null) {

			String clean_line = l.replaceAll(",", "."); // In the dataset ','
														// represents a '.'
			String[] cols = clean_line.split(";"); // Splitting the lines column
													// wise.
			if (cols.length == 0)
				break; // Ignoring empty lines
			if (cols[0].contains("Date"))
				continue; // Ignoring the description lines
			// If the temperature after 2 days is not present, break the loop.
			if (labels[ind2] == null)
				break;

			// In the dataset '-200' represents the default value if the value
			// is not present.
			if (cols[12].contains("-200") || labels[ind2].contains("-200")) {
				ind2 = ind2 + 1;
				continue;
			}

			// Diff stores the one day temperature difference at the same time
			// like 10/3 - 11/3 at 06:00 pm
			double diff;
			if (ind2 - 48 >= 0) {
				diff = Double.parseDouble(cols[12]) - Double.parseDouble(labels[ind2 - 48]);
				// if (ind2 - 48 == 0) System.out.println("&&& "+cols[12]+" , "+
				// labels[ind2-48]);
			} else {
				diff = 0.0;
			}

			// Writing the diff in first column then the full cleaned line and
			// at the list column we will write the temperature.
			bw.write(diff + ";" + clean_line + labels[ind2] + "\n");
			if (ind2 == 48)
				System.out.println(diff + ";" + clean_line + labels[ind2] + "\n");
			ind2 = ind2 + 1;
		}

		// *****************SPARK**************************************

		SparkConf conf = new SparkConf().setAppName("Weather Prediction");

		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<String> data = sc.textFile(clenedDataFile).cache();

		// 18 is the total # of columns in our new file.
		JavaRDD<String> filteredData = data.filter(x -> x.split(";").length == 18);

		JavaRDD<LabeledPoint> parsedData = filteredData.map(line -> {

			String[] parts = line.split(";");
			// We are going to use value at index 0: Temp difference and at
			// index 13: The temperature value.
			// Sample example of the value pattern:
			// At index 0 temperature difference between 10/3 and 11/3 at 6pm.
			// Then at index 13 the value will be actual temp on 11/3 at 6pm.
			// I am using index 0 and index 13 as my features for the
			// prediction.
			String[] features = { parts[0], parts[13] };

			double[] v = new double[features.length];
			for (int i = 0; i < features.length; i++) {
				v[i] = Double.parseDouble(features[i]);
			}

			// Taking the same sample example of the value pattern:
			// At index 0 temperature difference between 10/3 and 11/3 at 6pm.
			// Then at index 13 the value will be actual temp on 11/3 at 6pm.
			// The value at index 17 will be Temperature value on 12/3 at 6pm.
			// I am using index 17 as my Label in Logistic regression algorithm.
			return new LabeledPoint(Double.parseDouble(parts[17]), Vectors.dense(v));

		});

		parsedData.cache();

		// Splitting the data: 70% for training and remaining 30% for Testing.
		JavaRDD<LabeledPoint>[] tmp = parsedData.randomSplit(new double[] { 0.7, 0.3 });

		// Training set
		JavaRDD<LabeledPoint> training = tmp[0];
		// Test set
		JavaRDD<LabeledPoint> test = tmp[1];

		// Building the model
		int numIterations = 100;
		double stepSize = 0.001;	//Learning rate 'Alpha' for Stochastic Gradient Descent Algorithm

		LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(training), numIterations, stepSize);

		// Evaluate model on training examples and compute training error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds_train = training.map(

				new Function<LabeledPoint, Tuple2<Double, Double>>() {

					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<>(prediction, point.label());
					}

				});

		// valuesAndPreds_train.take(5);
		// System.out.println(valuesAndPreds_train.take(5).get(1)); //(predicted
		// value, Label)

		// Let's calculate mean squared training error
		double MSE_train = new JavaDoubleRDD(valuesAndPreds_train.map(new Function<Tuple2<Double, Double>, Object>() {

			public Object call(Tuple2<Double, Double> pair) {
				System.out.println(" " + pair._1() + " " + pair._2());
				return Math.pow(pair._1() - pair._2(), 2.0);
			}

		}).rdd()).mean();

		System.out.println("Train Mean Squared Error = " + MSE_train);

		// Evaluate model on testing examples and compute testing error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
			public Tuple2<Double, Double> call(LabeledPoint point) {
				double prediction = model.predict(point.features());
				return new Tuple2<>(prediction, point.label());
			}
		});

		// Let's compute mean testing error.
		double MSE = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {
			public Object call(Tuple2<Double, Double> pair) {
				return Math.pow(pair._1() - pair._2(), 2.0);
			}
		}).rdd()).mean();

		System.out.println("Test Mean Squared Error = " + MSE);
		sc.stop();
		br_l.close();
		br.close();
		bw.close();
	}
}