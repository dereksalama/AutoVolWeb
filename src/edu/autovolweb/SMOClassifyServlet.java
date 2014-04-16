package edu.autovolweb;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.KDTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import com.google.gson.JsonObject;

/**
 * Servlet implementation class SMOClassifyServlet
 */
@WebServlet("/SMOClassifyServlet")
public class SMOClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 4544453414238079731L;
	private Instances clusteredData;
	private Instances rawData;
	private Instances locData;
	private KDTree clusteredKd;
	private KDTree rawKd;
	
	private FilteredClusterer locClusterer;
	private Set<String> locClusters;
	
	private Instance createTarget(Instances data, HttpServletRequest req) throws Exception {
    	Instance target = new DenseInstance(data.numAttributes());
    	target.setDataset(data);
    	
    	Map<String, String[]> attrMap = req.getParameterMap();
    	for (Entry<String, String[]> e : attrMap.entrySet()) {
    		Attribute attr = data.attribute(e.getKey());
    		if (attr == null || attr.name().equals("lat")
    				|| attr.name().equals("lat")
    				|| attr.name().equals("long") // TODO
    				|| attr.name().equals("loc_provider")) {
    			continue;
    		}
    		Double val = Double.valueOf(e.getValue()[0]);
    		target.setValue(attr, val);
    	}
    	
    	Instance locInstance = new DenseInstance(3);
    	locInstance.setDataset(locData);
    	double lat = Double.valueOf(req.getParameter("lat"));
    	double lon = Double.valueOf(req.getParameter("long"));
    	locInstance.setValue(locData.attribute("lat"), lat);
    	locInstance.setValue(locData.attribute("lon"), lon);
    	
    	String locProvider = req.getParameter("loc_provider");
    	locInstance.setValue(locData.attribute("loc_provider"), locProvider);

  
    	String assignment = "" + locClusterer.clusterInstance(locInstance);
    	Attribute loc = data.attribute("loc");
    	String locClusterResult = "other";
    	if (locClusters.contains(assignment)) {
    		locClusterResult = assignment;
    	}
    	target.setValue(loc, locClusterResult);
    	
    	// replace missing values
    	for (int i = 0; i < target.numAttributes(); i++) {
    		if (target.isMissing(i)) {
    			double meanOrMode = data.meanOrMode(i);
    			target.setValue(i, meanOrMode);
    		}
    	}
    	
    	return target;
	}
	
    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws IOException {
    	

		try {
	    	JsonObject json = new JsonObject();
	        resp.setContentType("text/plain");
	        
	        Instance clusteredTarget = createTarget(clusteredData, req);
	        
			String kdThree = kdClassify(clusteredData, clusteredKd, clusteredTarget, 3);
			json.addProperty("k3", kdThree);
			
			String kdFive = kdClassify(clusteredData, clusteredKd, clusteredTarget, 7);
			json.addProperty("k7", kdFive);
			
			Instance rawTarget = createTarget(rawData, req);
			String rawKdThree = kdClassify(rawData, rawKd, rawTarget, 3);
			json.addProperty("raw_k3", rawKdThree);
			
			String rawKdFive = kdClassify(rawData, rawKd, rawTarget, 7);
			json.addProperty("raw_k7", rawKdFive);
			
			json.addProperty("loc", rawTarget.value(rawData.attribute("loc")));
			
		    resp.getWriter().write(json.toString());
		} catch (Exception e1) {
			e1.printStackTrace();
		}

    }
    
    private String kdClassify(Instances data, KDTree kd, Instance target, int k) {
    	String result = "err";
		try {
			Instances neighbors = kd.kNearestNeighbours(target, k);
			int numClassValues = data.classAttribute().numValues();
			int[] classCounts = new int[numClassValues];
			for (Instance n : neighbors) {
				classCounts[(int) Math.round(n.classValue())]++;
			}
			
			int maxIndex = 0;
			int maxCount = Integer.MIN_VALUE;
			for (int i = 0; i < numClassValues; i++) {
				if (classCounts[i] > maxCount) {
					maxCount = classCounts[i];
					maxIndex = i;
				}
			}
			
			return data.classAttribute().value(maxIndex);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    	return result;
    }
    
    @Override
    public void init() throws ServletException {
    	Instances clusteredDataNoLoc = loadDataFile("WEB-INF/resources/labeled_output.csv");
    	Instances rawDataWithTs = loadDataFile("WEB-INF/resources/output.csv");

		try {
			locData = CurrentStateUtil.extractLocationData(clusteredDataNoLoc, true);
			locClusterer = CurrentStateUtil.trainLocationClusterer(locData, 20);
			List<String> topClusterList = CurrentStateUtil.findTopClusters((SimpleKMeans) locClusterer.getClusterer(), 
					locData.numInstances());
			locClusters = new HashSet<>(topClusterList.size());
			locClusters.addAll(topClusterList);
					
	    	Remove removeTs = new Remove();
	    	removeTs.setAttributeIndicesArray(new int[] {0});
	    	removeTs.setInputFormat(rawDataWithTs);
	    	Instances rawDataNoLoc = Filter.useFilter(rawDataWithTs, removeTs);
	    	Instances rawDataMissingVals = CurrentStateUtil.replaceLocationData(rawDataNoLoc, 
	    			new int[]{4, 5, 6},
	    			topClusterList, 
	    			((SimpleKMeans)locClusterer.getClusterer()).getAssignments());
	    	
	    	ReplaceMissingValues rmv = new ReplaceMissingValues();
	    	rmv.setInputFormat(rawDataMissingVals);
	    	rawData = Filter.useFilter(rawDataMissingVals, rmv);
	    	// hack to make ringer nominal
	    	double[] ringerVals = rawData.attributeToDoubleArray(rawData.attribute("ringer").index());
	    	rawData.delete(rawData.attribute("ringer").index());
	    	List<String> ringerTypes = new ArrayList<>();
	    	ringerTypes.add("silent");
	    	ringerTypes.add("vibrate");
	    	ringerTypes.add("normal");
	    	Attribute newRinger = new Attribute("new_ringer", ringerTypes);
	    	rawData.insertAttributeAt(newRinger, rawData.numAttributes());
	    	for (int i = 0; i < rawData.numInstances(); i++) {
	    		String ringer = "err";
	    		double val = ringerVals[i];
	    		if (val == 0.0) {
	    			ringer = "silent";
	    		} else if (val == 1.0) {
	    			ringer = "vibrate";
	    		} else if (val == 2.0) {
	    			ringer = "normal";
	    		}
	    		rawData.instance(i).setValue(rawData.attribute("new_ringer"), ringer);
	    	}
	    	rawData.setClass(rawData.attribute("new_ringer"));
	    	
	    	clusteredData = CurrentStateUtil.replaceLocationData(clusteredDataNoLoc, 
	    			new int[]{2, 3, 4},
	    			topClusterList, 
	    			((SimpleKMeans)locClusterer.getClusterer()).getAssignments());
	    	clusteredData.setClassIndex(clusteredData.numAttributes() - 1);
	    	
	    	clusteredKd = new KDTree();
			clusteredKd.setInstances(clusteredData);
			
			rawKd = new KDTree();
			rawKd.setInstances(rawData);
		} catch (Exception e) {
			e.printStackTrace();
			throw new ServletException("Error building classifier");
		}
    }

	private Instances loadDataFile(String filename) throws ServletException {
		InputStream dataFileStream = getServletContext()
    			.getResourceAsStream(filename);
		Instances data = null;

    	CSVLoader loader = new CSVLoader();
    	try {
			loader.setSource(dataFileStream);
			data = loader.getDataSet();
			data.setClassIndex(data.numAttributes() - 1);
			dataFileStream.close();
		} catch (IOException e) {
			e.printStackTrace();
			throw new ServletException("Unable to load csvfile: " + filename);
		}

    	return data;
	}


}
