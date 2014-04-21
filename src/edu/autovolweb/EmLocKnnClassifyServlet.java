package edu.autovolweb;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;

import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * Servlet implementation class EmLocKnnClassifyServlet
 */
@WebServlet("/EmLocKnnClassifyServlet")
public class EmLocKnnClassifyServlet extends BaseKnnClassify {
	private static final long serialVersionUID = 1L;
	
	private static final String EM_CLASSIFIER_FILE = "loc_em_classifier";
	
	private Map<String, EM> emMap;
	private Map<String, List<String>> clusterMap;
	
	@Override
	public void init() throws ServletException {
		super.init();
		emMap = new HashMap<>();
		clusterMap = new HashMap<>();
	}

	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		
		String filename = DataUploadServlet.constructUserFileName(userId, EM_CLASSIFIER_FILE);
		File f = new File(filename);
		if (!f.exists()) {
			throw new Exception("Loc EM not ready");
		}
		EM em = (EM) SerializationHelper.read(new FileInputStream(filename));
		emMap.put(userId, em);
		
		return replaceLocData(userId, data, em);
	}
	
	@Override
	protected String prepareOutput(JsonObject json, Instance target) {
		json.addProperty("loc", target.value(target.dataset().attribute("loc")));
		json.addProperty("prop_loc", target.value(target.dataset().attribute("prob_loc")));
		
		return json.toString();
	}

	@Override
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		Instance locInst = extractLocInstance(state);
		
		EM em = emMap.get(userId);
		try {
			int cluster = em.clusterInstance(locInst);
			double[] probs = em.distributionForInstance(locInst);
			double prob = probs[cluster];
			
			Instance i = CurrentStateUtil.toEmLocInstance(state, "" + cluster,
					prob, clusterMap.get(userId));
			
			return i;
		} catch (Exception e) {
			e.printStackTrace();
		}

		
		return null;
	}
	
	public static class ClusterLocations implements Runnable {
		private final Instances input;
		private final String userId;
		
		public ClusterLocations(Instances input, String userId) {
			this.input = input;
			this.userId = userId;
		}

		@Override
		public void run() {
			String filename = DataUploadServlet.constructUserFileName(userId, EM_CLASSIFIER_FILE);
			File f = new File(filename);
			if (f.exists()) {
				f.delete();
			}
			EM em = new EM();
			try {
				Instances locData = extractLocationData(input);
				em.buildClusterer(locData);
				
				SerializationHelper.write(new FileOutputStream(filename), em);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	private Instances replaceLocData(String userId, Instances data, EM em) throws Exception {
		ArrayList<String> clusterNames = new ArrayList<String>(em.getNumClusters());
		for (int i = 0; i < em.getNumClusters(); i++) {
			clusterNames.add(i + "");
		}
		clusterMap.put(userId, clusterNames);
		
		Remove removeLoc = new Remove();
		removeLoc.setAttributeIndicesArray(new int[]{2,3,4});
		removeLoc.setInputFormat(data);
		
		Instances result = Filter.useFilter(data, removeLoc);
		Attribute locAttr = new Attribute("loc", clusterNames);
		result.insertAttributeAt(locAttr, result.numAttributes() - 1);
		Attribute locProbAttr = new Attribute("prob_loc");
		result.insertAttributeAt(locProbAttr, result.numAttributes() - 1);
		
		
		Instances locData = extractLocationData(data);
		for (int i = 0; i < data.numInstances(); i++) {
			Instance locInst = locData.get(i);
			int cluster = em.clusterInstance(locInst);
			double[] probs = em.distributionForInstance(locInst);
			double prob = probs[cluster];
			result.instance(i).setValue(result.attribute("loc"), cluster + "");
			result.instance(i).setValue(result.attribute("prob_loc"), prob);
		}
		
		return result;
	}
	
	private Instance extractLocInstance(CurrentStateData state) {
		Instance locInstance = new DenseInstance(2);
		
		ArrayList<Attribute> locAttr = new ArrayList<>();
		locAttr.add(new Attribute("lat"));
		locAttr.add(new Attribute("lon"));
		
		Instances locDataset = new Instances("loc_data", locAttr, 0);
		locInstance.setDataset(locDataset);
		
		locInstance.setValue(locDataset.attribute("lat"), state.getLat());
		locInstance.setValue(locDataset.attribute("lon"), state.getLon());
		
		return locInstance;
	}
	
	private static Instances extractLocationData(Instances input) {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("lat"));
		attributes.add(new Attribute("lon"));

		Instances locData = new Instances("loc_data", attributes, input.numInstances());
		
		for (Instance orig : input) {
			Instance loc = new DenseInstance(2);
			loc.setDataset(locData);
			
			loc.setValue(locData.attribute("lat"), orig.value(input.attribute("lat")));
			loc.setValue(locData.attribute("lon"), orig.value(input.attribute("lon")));
			locData.add(loc);
		}
		
		return locData;
	}

}
