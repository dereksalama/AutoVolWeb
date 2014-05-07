package edu.autovolweb;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;

import weka.clusterers.EM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import com.google.gson.Gson;

/**
 * Servlet implementation class EmLocRfClassifyServlet
 */
@WebServlet("/EmLocRfClassifyServlet")
public class EmLocRfClassifyServlet extends RfClassifyServlet {
	private static final long serialVersionUID = 1L;

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
		
		String filename = DataUploadServlet.constructUserFileName(userId, 
				EmLocKnnClassifyServlet.EM_CLASSIFIER_FILE);
		File f = new File(filename);
		if (!f.exists()) {
			throw new Exception("Loc EM not ready");
		}
		EM em = (EM) SerializationHelper.read(new FileInputStream(filename));
		emMap.put(userId, em);
		
		ArrayList<String> clusterNames = new ArrayList<String>(em.numberOfClusters());
		for (int i = 0; i < em.numberOfClusters(); i++) {
			clusterNames.add(i + "");
		}
		clusterMap.put(userId, clusterNames);
		
		return EmLocKnnClassifyServlet.replaceLocData(data, em, clusterNames);
	}
	

	@Override
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		Instance locInst = EmLocKnnClassifyServlet.extractLocInstance(state);
		
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
	
	@Override
	protected void clear() {
		super.clear();
		emMap.clear();
		clusterMap.clear();
	}

}
