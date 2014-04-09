package edu.autovolweb;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Servlet implementation class ViewDataServlet
 */
@WebServlet("/ViewDataServlet")
public class ViewDataServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;

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
		List<Instances> allInstances = new ArrayList<Instances>();
		DateTime today = new DateTime();
		// load previous data (up to how old?)
		for (int i = 0; i < DataUploadServlet.DATA_AGE; i++) {
			DateTime day = today.minusDays(i);
			String fileName = DataUploadServlet.constructArffFileName(day, userId, this);
			File f = new File(fileName);
			if (f.exists()) {
				try {
					//BufferedReader reader = new BufferedReader(new FileReader(f));
					//Instances moreData = new Instances(reader);
					ArffLoader loader = new ArffLoader();
					loader.setFile(f);
					Instances moreData = loader.getDataSet();
					allInstances.add(moreData);
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			} else {
				break; // have gone past oldest file
			}
		}
		
		if (allInstances.isEmpty()) {
			response.getWriter().write("No data found");
			return;
		}
		
		Instances allData = allInstances.get(0);
		allData.setClassIndex(allData.numAttributes() - 1);
		allInstances.remove(0);
		for (Instances i : allInstances) {
			allData.addAll(i);
		}
		allData.sort(1);
		
		String locClustersParam = request.getParameter("k");
		int numLocClusters = Integer.valueOf(locClustersParam);
		
		//Load classifier
		try {
			/*
			FilteredClusterer em = (FilteredClusterer) SerializationHelper.read(new FileInputStream(
					DataUploadServlet.EM_MODEL_FILE));

			Remove remove = new Remove();
			remove.setAttributeIndices("" + (allData.classIndex() + 1));
			remove.setInputFormat(allData);

			Instances unlabeledData = Filter.useFilter(allData, remove);
			 */
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

			for (int i = 0; i < allDataLoc.numInstances(); i++) {
				Instance target = unlabeledData.get(i);
				double[] distrib = em.distributionForInstance(target);
				int maxCluster = em.clusterInstance(target);
				EMCluster cluster = clusterToLabels.get(maxCluster);
				String label = cluster.getRingerLabel();
				double probOfCluster = distrib[maxCluster];
				double probOfLabel = cluster.getProbOfLabel();

				response.getWriter().write(
						allDataLoc.get(i) + "\n" +
								"\t label: " + label + " cluster: " + maxCluster + 
								" probOfCluster: " + probOfCluster
								+ " probOfLabel: " + probOfLabel + "\n\n");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
