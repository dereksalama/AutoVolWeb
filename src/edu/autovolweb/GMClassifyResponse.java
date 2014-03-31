package edu.autovolweb;

public class GMClassifyResponse {
	private String ringer;
	private Double probOfLabel;
	private Double probOfCluster;
	
	public GMClassifyResponse() {}
	public GMClassifyResponse(String ringer, Double probOfLabel, Double probOfCluster) {
		this.ringer = ringer;
		this.probOfCluster = probOfCluster;
		this.probOfLabel = probOfLabel;
	}
	
	public String getRinger() {
		return ringer;
	}

	public Double getProbOfLabel() {
		return probOfLabel;
	}

	public Double getProbOfCluster() {
		return probOfCluster;
	}

}
