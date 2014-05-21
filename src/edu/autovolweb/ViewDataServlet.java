package edu.autovolweb;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.joda.time.DateTime;

import weka.clusterers.EM;
import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/**
 * Servlet implementation class ViewDataServlet
 */
@WebServlet("/ViewDataServlet")
public class ViewDataServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	
	private static String nameOfLogger = ViewDataServlet.class.getName();
	private static Logger logger = Logger.getLogger(nameOfLogger);
	
	static class TimeComparator implements Comparator<Instance> {

		@Override
		public int compare(Instance lhs, Instance rhs) {
			int dayIndex = lhs.dataset().attribute("day").index();
			int timeIndex =  lhs.dataset().attribute("time").index();
			if (lhs.value(dayIndex) > rhs.value(dayIndex)) {
				return -1;
			} else if (lhs.value(dayIndex) < rhs.value(dayIndex)) {
				return 1;
			}
			
			if (lhs.value(timeIndex) > rhs.value(timeIndex)) {
				return -1;
			} else if (lhs.value(timeIndex) < rhs.value(timeIndex)) {
				return 1;
			}
			
			return 0;
		}
		
	}
	
	public static Instances loadAllData(String userId, HttpServlet servlet) { // TODO: remove servlet param
		List<Instances> allInstances = new ArrayList<Instances>();
		DateTime today = new DateTime();
		for (int i = DataUploadServlet.DATA_AGE - 1; i >= 0; i--) {
			DateTime day = today.minusDays(i);
			String fileName = DataUploadServlet.constructArffFileName(day, userId);
			File f = new File(fileName);
			if (f.exists()) {
				try {
					ArffLoader loader = new ArffLoader();
					loader.setFile(f);
					Instances moreData = loader.getDataSet();
					//Collections.sort(moreData, new TimeComparator());
					allInstances.add(moreData);
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					logger.log(Level.SEVERE, "Error loading file " + fileName);
					e.printStackTrace();
				}
			}
		}
		
		if (allInstances.isEmpty()) {
			return null;
		}
		
		Instances allDataWifi = allInstances.get(0);
		Instances allData = removeWifiCount(allDataWifi);
		allData.setClass(allData.attribute("ringer"));
		allInstances.remove(0);
		for (Instances i : allInstances) {
			if (i != null) {
				i = removeWifiCount(i);
				allData.addAll(i);
			}
		}
		//allData.sort(1);
		
		return allData;
	}

	private static Instances removeWifiCount(Instances allData) {
		Remove r = new Remove();
		int[] attrIndices = new int[1];
		if (allData.attribute("wifi_count") == null) {
			return allData;
		}
		attrIndices[0] = allData.attribute("wifi_count").index();
		r.setAttributeIndicesArray(attrIndices);
		try {
			r.setInputFormat(allData);
			Instances result = Filter.useFilter(allData, r);
			return result;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	private void viewData(Instances allData, PrintWriter writer) {
		writer.write("total: " + allData.numInstances() + "\n");
		for (Instance i : allData) {
			writer.write(i.toString());
			writer.write("\n");
		}
	}
	
	private String getFileJson(String userId, String filename) throws IOException {
		String fullFilename = DataUploadServlet.constructUserFileName(userId, filename);
		byte[] encoded = Files.readAllBytes(Paths.get(fullFilename));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		return json;
	}

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// Load cluster label mapping
		/*
		byte[] encoded = Files.readAllBytes(Paths.get(
				DataUploadServlet.CLUSTER_LABELS_FILE));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		
		Gson gson = new Gson();
		Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
		List<EMCluster> clusters = gson.fromJson(json, collectionType);
		*/
		String userId = request.getParameter("user");
		Instances allData = loadAllData(userId, this);
		if (allData == null) {
			response.getWriter().write("no data");
			return;
		}
		
		String type = request.getParameter("type");
		if (type == null) {
			type = "view";
		}
		
		if (type.equals("view")) {
			viewData(allData, response.getWriter());
			return;
		} else if (type.equals("loc")) {
			SimpleKMeans locClusterer;
			try {
				locClusterer = (SimpleKMeans) SerializationHelper.read(new FileInputStream(
						DataUploadServlet.constructUserFileName(userId,DataUploadServlet.LOC_CLUSTERER_FILE)));
				
				String locJson = getFileJson(userId, DataUploadServlet.LOC_DATA_FILE);
				Type locCollectionType = new TypeToken<List<String>>(){}.getType();
				
				Gson gson = new Gson();
				List<String> topClusterList = gson.fromJson(locJson, locCollectionType);
				Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
						new int[] {2,3,4}, topClusterList, 
						locClusterer.getAssignments());
				allDataLoc.setClass(allDataLoc.attribute("ringer"));
				viewData(allDataLoc, response.getWriter());
			} catch (Exception e) {
				e.printStackTrace(response.getWriter());
			}
			return;
		} else if (type.equals("prob_loc")) {
			
			try {
				EM em = (EM) SerializationHelper.read(new FileInputStream(
						DataUploadServlet.constructUserFileName(userId, EmLocKnnClassifyServlet.EM_CLASSIFIER_FILE)));
				
				
				ArrayList<String> clusterNames = new ArrayList<String>(em.numberOfClusters());
				for (int i = 0; i < em.numberOfClusters(); i++) {
					clusterNames.add(i + "");
				}
				Instances allDataLoc = EmLocKnnClassifyServlet.replaceLocData(allData, em,
						clusterNames);
				
				response.getWriter().write("num loc clusters: " + em.numberOfClusters() + "\n");
				allDataLoc.setClass(allDataLoc.attribute("ringer"));
				viewData(allDataLoc, response.getWriter());
			} catch (Exception e) {
				e.printStackTrace(response.getWriter());
			}
			return;
			
		} else if (type.equals("load")) {
			try {
				FilteredClusterer em = (FilteredClusterer) SerializationHelper.read(new FileInputStream(
						DataUploadServlet.EM_MODEL_FILE));
				
				String clusterLabelsJson = getFileJson(userId, 
						DataUploadServlet.CLUSTER_LABELS_FILE);
				
				Gson gson = new Gson();
				Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
				List<EMCluster> clusterToLabels = gson.fromJson(clusterLabelsJson, collectionType);
				
				SimpleKMeans locClusterer = (SimpleKMeans) SerializationHelper.read(new FileInputStream(
						DataUploadServlet.constructUserFileName(userId,DataUploadServlet.LOC_CLUSTERER_FILE)));
				
				String locJson = getFileJson(userId, DataUploadServlet.LOC_DATA_FILE);
				Type locCollectionType = new TypeToken<List<String>>(){}.getType();
				List<String> topClusterList = gson.fromJson(locJson, locCollectionType);
				Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
						new int[] {2,3,4}, topClusterList, 
						locClusterer.getAssignments());
				allDataLoc.setClass(allDataLoc.attribute("ringer"));
				
				Remove removeClass = new Remove();
				removeClass.setAttributeIndices("" + (allDataLoc.classIndex() + 1));
				removeClass.setInputFormat(allDataLoc);
				Instances unlabeledData = Filter.useFilter(allDataLoc, removeClass);
				
				
				viewWithCluster(response.getWriter(), allDataLoc, unlabeledData, em, 
						clusterToLabels);
				return;
			} catch (Exception e) {
				e.printStackTrace();
				response.getWriter().write("error loading models\n");
				e.printStackTrace(response.getWriter());
				return;
			}
		} else if (type.equals("avg")) {
			Instances avgData = AvgKnnClassifyServlet.avgData(allData, 4);
			
			viewData(avgData, response.getWriter());
			return;
		}

		int numLocClusters = Integer.valueOf(request.getParameter("k"));

		
		//Load classifier
		try {

			Instances locData = CurrentStateUtil.extractLocationData(allData, false);
			SimpleKMeans locClusterer = CurrentStateUtil.trainUnfilteredLocationClusterer(locData, 
					numLocClusters);
			List<String> topClusterList = CurrentStateUtil.findTopClusters(locClusterer, 
					allData.numInstances());
			Set<String> locClusters = new HashSet<>(topClusterList.size());
			locClusters.addAll(topClusterList);
			
			Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
					new int[] {2,3,4}, topClusterList, 
					locClusterer.getAssignments());
			allDataLoc.setClass(allDataLoc.attribute("ringer"));
			
			
			EM unfilteredEM = new EM();
			unfilteredEM.setMaximumNumberOfClusters(20);
			Normalize normalizer = new Normalize();

			Remove removeClass = new Remove();
			removeClass.setAttributeIndices("" + (allDataLoc.classIndex() + 1));
			removeClass.setInputFormat(allDataLoc);
			Instances unlabeledData = Filter.useFilter(allDataLoc, removeClass);

			normalizer.setInputFormat(unlabeledData);
			FilteredClusterer em = new FilteredClusterer();
			em.setClusterer(unfilteredEM);
			em.setFilter(normalizer);
			em.buildClusterer(unlabeledData);
			List<EMCluster> clusterToLabels = EMCluster
					.createClusterToLabelMap(allDataLoc, unlabeledData, em);

			viewWithCluster(response.getWriter(), allDataLoc, unlabeledData, em,
					clusterToLabels);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void viewWithCluster(PrintWriter writer,
			Instances allDataLoc, Instances unlabeledData,
			FilteredClusterer em, List<EMCluster> clusterToLabels)
			throws Exception, IOException {
		for (int i = 0; i < allDataLoc.numInstances(); i++) {
			Instance target = unlabeledData.get(i);
			double[] distrib = em.distributionForInstance(target);
			int maxCluster = em.clusterInstance(target);
			EMCluster cluster = clusterToLabels.get(maxCluster);
			String label = cluster.getRingerLabel();
			double probOfCluster = distrib[maxCluster];
			double probOfLabel = cluster.getProbOfLabel();

			writer.write(
					allDataLoc.get(i) + "\n" +
							"\t label: " + label + " cluster: " + maxCluster + 
							" probOfCluster: " + probOfCluster
							+ " probOfLabel: " + probOfLabel + "\n\n");
		}
	}
}
