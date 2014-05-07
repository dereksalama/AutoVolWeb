package edu.autovolweb;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

import com.google.gson.JsonObject;

public abstract class BaseKnnClassify extends ClearingHttpServlet {
	private static final long serialVersionUID = -1119373032862346060L;

	private Map<String, KDTree> kdMap;
	private Set<String> initializedUsers;
	
	private static int[] ks = new int[] {3, 7, 13};
	
	@Override
	protected void clear() {
		kdMap.clear();
		initializedUsers.clear();	
	}

	
	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		if (userId == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		if (!initializedUsers.contains(userId)) {
			try {
				initForUser(userId);
			} catch (Exception e) {
				e.printStackTrace();
				response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
				return;
			}
			
		}
		initializedUsers.add(userId);
		String input = request.getParameter("target");
		if (input == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
		
		Instance target = constructTarget(input, userId);
		if (target == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		JsonObject json = new JsonObject();
		KDTree kd = kdMap.get(userId);
		for (int k : ks) {
			addResult(json, target, kd, k);
		}
		
		response.getWriter().write(prepareOutput(json, target));
	}
	
	protected String prepareOutput(JsonObject json, Instance target) {
		return json.toString();
	}
	
	protected Instances getDataset(String userId) {
		KDTree kd = kdMap.get(userId);
		return kd.getInstances();
	}
	
	@Override
	public void init() throws ServletException {
		kdMap = new HashMap<>();
		initializedUsers = Collections.synchronizedSet(new HashSet<String>());
	}
	
	protected void initForUser(String userId) throws Exception {
		Instances data = loadData(userId);
		data.setClass(data.attribute("ringer"));
		KDTree kd = new KDTree();
		kdMap.put(userId, kd);
		
		kd.setInstances(data);
	}
	
   private int[] kdClassify(Attribute classAttr, KDTree kd, Instance target, int k) {
		try {
			Instances neighbors = kd.kNearestNeighbours(target, k);
			int numClassValues = classAttr.numValues();
			int[] classCounts = new int[numClassValues];
			for (Instance n : neighbors) {
				classCounts[(int) Math.round(n.classValue())]++;
			}
			return classCounts;
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    	return null;
    }
   
    private void addResult(JsonObject json, Instance target, KDTree kd, int k) {
    	int[] classCounts = kdClassify(target.classAttribute(), kd, target, k);
    	
    	int maxIndex = 0;
		int maxCount = Integer.MIN_VALUE;
		for (int i = 0; i < target.classAttribute().numValues(); i++) {
			if (classCounts[i] > maxCount) {
				maxCount = classCounts[i];
				maxIndex = i;
			}
		}
		
		String label = target.classAttribute().value(maxIndex);
		//Double prob = ((double) maxCount) / k;
		json.addProperty("" + k, label);
		//json.addProperty(k + "_prob", prob);
    }
	
	protected abstract Instances loadData(String userId) throws Exception;
	
	protected abstract Instance constructTarget(String input, String userId);
}
