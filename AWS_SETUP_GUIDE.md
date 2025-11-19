# AWS EMR Setup Guide

This guide will help you set up AWS infrastructure for the fraud detection project, including fixing the **PassRole permission error** you're encountering in EMR Studio.

## Current Issue: PassRole Permission Error

The error you're seeing: `"HTTP 403: Forbidden (Authorization Error: User does not have the PassRole permission for the execution role.)"` means your IAM user/role doesn't have permission to pass IAM roles to EMR.

---

## Step 1: AWS Account Setup

### 1.1 AWS CLI Installation and Configuration

```bash
# Install AWS CLI (if not installed)
# Windows: Download from https://aws.amazon.com/cli/
# Or use: pip install awscli

# Configure AWS CLI
aws configure
```

**Enter:**

- AWS Access Key ID: [Your access key from AWS IAM]
- AWS Secret Access Key: [Your secret key]
- Default region: `us-east-1` (or `us-east-2` if that's what you're using)
- Default output format: `json`

### 1.2 Verify Configuration

```bash
aws sts get-caller-identity
```

You should see your AWS account ID and user ARN.

---

## Step 2: IAM Setup (CRITICAL - Fixes PassRole Error)

### 2.1 Create IAM Role for EMR Service

1. **Go to AWS Console → IAM → Roles → Create Role**
2. **Select trusted entity:**
   - Choose "AWS Service"
   - Select "EMR" or "Elastic MapReduce"
3. **Attach Policies:**

   - `AmazonElasticMapReduceFullAccess`
   - `AmazonS3FullAccess` (or create custom policy with only needed S3 permissions)
   - `CloudWatchFullAccess` (for logging)

4. **Name the role:** `EMR_DefaultRole` (or your preferred name)
5. **Note the Role ARN** (you'll need this)

### 2.2 Create Instance Profile Role

1. **Create another role for EC2 instances:**

   - Same steps, but choose "EC2" as trusted entity
   - Attach: `AmazonS3FullAccess`, `CloudWatchFullAccess`
   - Name: `EMR_EC2_DefaultRole`

2. **Create Instance Profile:**
   ```bash
   aws iam create-instance-profile --instance-profile-name EMR_EC2_DefaultRole
   aws iam add-role-to-instance-profile \
       --instance-profile-name EMR_EC2_DefaultRole \
       --role-name EMR_EC2_DefaultRole
   ```

### 2.3 Fix PassRole Permission (SOLVES YOUR ERROR)

Your IAM user needs permission to pass these roles to EMR. Create a custom policy:

1. **Go to IAM → Policies → Create Policy**
2. **Use JSON editor:**

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["iam:PassRole", "iam:GetRole", "iam:ListRoles"],
         "Resource": [
           "arn:aws:iam::YOUR_ACCOUNT_ID:role/EMR_DefaultRole",
           "arn:aws:iam::YOUR_ACCOUNT_ID:role/EMR_EC2_DefaultRole",
           "arn:aws:iam::YOUR_ACCOUNT_ID:role/*EMR*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": [
           "emr:CreateCluster",
           "emr:DescribeCluster",
           "emr:TerminateCluster",
           "emr:ListClusters",
           "emr:RunJobFlow"
         ],
         "Resource": "*"
       },
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::YOUR_BUCKET_NAME/*",
           "arn:aws:s3:::YOUR_BUCKET_NAME"
         ]
       }
     ]
   }
   ```

   **Replace:**

   - `YOUR_ACCOUNT_ID` with your AWS account ID (found in `aws sts get-caller-identity`)
   - `YOUR_BUCKET_NAME` with your S3 bucket name (or use `*` for all buckets)

3. **Name the policy:** `EMR_FraudDetection_Policy`
4. **Attach to your IAM user:**
   - Go to IAM → Users → [Your User] → Add Permissions → Attach Policies
   - Select `EMR_FraudDetection_Policy`

### 2.4 EMR Studio Service Role (If using EMR Studio)

1. **Create role:**

   - Trusted entity: `EMR Studio`
   - Attach: `AmazonElasticMapReduceFullAccess`, `AmazonS3FullAccess`
   - Name: `EMRStudioServiceRole`

2. **Add PassRole for this role too:**
   Update the PassRole policy to include:
   ```json
   "arn:aws:iam::YOUR_ACCOUNT_ID:role/EMRStudioServiceRole"
   ```

---

## Step 3: S3 Bucket Setup

### 3.1 Create S3 Bucket

```bash
# Set bucket name (must be globally unique)
export BUCKET_NAME="csp554-fraud-detection-$(date +%s)"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Or for us-east-2 (if that's your region):
aws s3 mb s3://$BUCKET_NAME --region us-east-2
```

### 3.2 Set Environment Variable

```bash
# Windows PowerShell:
$env:S3_BUCKET=$BUCKET_NAME

# Windows CMD:
set S3_BUCKET=%BUCKET_NAME%

# Linux/Mac:
export S3_BUCKET=$BUCKET_NAME
```

### 3.3 Upload Dataset

```bash
# Navigate to project directory
cd "C:\Users\91981\Desktop\Personal Projects\BigData_FinalProject"

# Upload dataset
aws s3 cp creditcard_2023.csv/creditcard_2023.csv \
    s3://$BUCKET_NAME/raw-data/creditcard_2023.csv

# Verify upload
aws s3 ls s3://$BUCKET_NAME/raw-data/
```

### 3.4 Create S3 Folder Structure

```bash
# Create directories
aws s3api put-object --bucket $BUCKET_NAME --key processed/
aws s3api put-object --bucket $BUCKET_NAME --key models/
aws s3api put-object --bucket $BUCKET_NAME --key outputs/
aws s3api put-object --bucket $BUCKET_NAME --key profiling/
aws s3api put-object --bucket $BUCKET_NAME --key emr-logs/
```

---

## Step 4: EMR Cluster Setup

### 4.1 Create EMR Cluster (via AWS Console)

1. **Go to AWS Console → EMR → Create Cluster**

2. **Cluster Configuration:**

   - **Name:** `fraud-detection-cluster`
   - **Release:** `emr-6.15.0` (or latest)
   - **Applications:** Select `Spark`, `Hadoop`, `JupyterEnterpriseGateway`

3. **Instance Configuration:**

   - **Instance type:** `m5.large` (or `m5.xlarge` for better performance)
   - **Number of instances:** `3` (1 master, 2 core)
   - **EC2 key pair:** Select your key pair (create one in EC2 if needed)

4. **Security and Access:**

   - **EC2 instance profile:** `EMR_EC2_DefaultRole` (the instance profile you created)
   - **EMR role:** `EMR_DefaultRole` (the service role you created)

5. **Additional Settings:**

   - **Auto-terminate:** Enable (clusters terminate after idle time)
   - **Logging:** Enable, set S3 path: `s3://$BUCKET_NAME/emr-logs/`

6. **Create Cluster**

### 4.2 Create EMR Cluster (via CLI)

```bash
aws emr create-cluster \
    --name "fraud-detection-cluster" \
    --release-label emr-6.15.0 \
    --instance-type m5.large \
    --instance-count 3 \
    --applications Name=Spark Name=Hadoop Name=JupyterEnterpriseGateway \
    --ec2-attributes KeyName=YOUR_KEY_NAME,InstanceProfile=EMR_EC2_DefaultRole \
    --service-role EMR_DefaultRole \
    --log-uri s3://$BUCKET_NAME/emr-logs/ \
    --auto-terminate \
    --region us-east-1
```

**Replace:**

- `YOUR_KEY_NAME` with your EC2 key pair name

### 4.3 Get Cluster Details

```bash
# List clusters
aws emr list-clusters --cluster-states WAITING RUNNING

# Get cluster ID
CLUSTER_ID=$(aws emr list-clusters \
    --cluster-states WAITING RUNNING \
    --query "Clusters[0].Id" \
    --output text)

# Get master node DNS
aws emr describe-cluster --cluster-id $CLUSTER_ID \
    --query "Cluster.MasterPublicDnsName" \
    --output text
```

---

## Step 5: EMR Studio Setup

### 5.1 Create EMR Studio

1. **Go to AWS Console → EMR → Studios → Create Studio**

2. **Studio Configuration:**

   - **Name:** `fraud-detection-studio`
   - **Auth mode:** `IAM` or `SSO` (IAM is simpler for individual use)
   - **Default S3 location:** `s3://$BUCKET_NAME/emr-studio/`
   - **Service role:** `EMRStudioServiceRole` (the one you created earlier)
   - **Subnet:** Select a subnet in your VPC
   - **Security groups:** Select appropriate security groups

3. **Create Studio**

### 5.2 Fix EMR Studio PassRole Error

After creating the studio, you need to ensure your user can pass roles:

1. **Get Studio ID:**

   ```bash
   STUDIO_ID=$(aws emr list-studios --query "Studios[0].StudioId" --output text)
   ```

2. **Verify Studio Service Role:**

   - Go to Studios → Your Studio → Settings
   - Check that "Execution role" is set correctly
   - The execution role should have permissions to access S3 and run EMR jobs

3. **Update User Policy:**
   Make sure your IAM user has PassRole permission for the execution role used by EMR Studio.

### 5.3 Create Workspace

1. **In EMR Studio, click "Create Workspace"**
2. **Name:** `fraud-detection-workspace`
3. **Attach to cluster:** Select your EMR cluster (or leave as "Serverless")
4. **Click "Create"**

### 5.4 Upload Code to S3

```bash
# Upload source code
aws s3 sync src/ s3://$BUCKET_NAME/code/src/

# Upload dataset schema
aws s3 cp DATASET_SCHEMA.md s3://$BUCKET_NAME/code/
```

---

## Step 6: Verify Setup

### 6.1 Test S3 Access

```bash
# List bucket contents
aws s3 ls s3://$BUCKET_NAME/ --recursive

# Test write
echo "test" > test.txt
aws s3 cp test.txt s3://$BUCKET_NAME/test/
aws s3 rm s3://$BUCKET_NAME/test/test.txt
```

### 6.2 Test EMR Access

```bash
# List clusters
aws emr list-clusters

# Describe cluster
aws emr describe-cluster --cluster-id $CLUSTER_ID
```

### 6.3 Test EMR Studio

1. **Open EMR Studio workspace**
2. **Create a test notebook**
3. **In Compute panel, select your EMR cluster**
4. **Select execution role** (the one with PassRole permissions)
5. **Click "Attach"**
6. **Run a test cell:**
   ```python
   print("Hello from EMR!")
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.appName("Test").getOrCreate()
   print(f"Spark version: {spark.version}")
   ```

---

## Step 7: Quick Fix for Current PassRole Error

If you're currently stuck with the PassRole error in EMR Studio:

### Option 1: Use Instance Profile (Quick Fix)

1. In EMR Studio, go to Compute panel
2. For "Runtime role", try selecting **No runtime role** or use the cluster's default instance profile
3. This might work if your cluster already has proper S3 permissions

### Option 2: Request Admin to Add PassRole Permission

If you don't have IAM admin access, ask your AWS administrator to:

1. Add `iam:PassRole` permission to your IAM user
2. Allow passing the EMR Studio execution role

### Option 3: Use SSH Instead of EMR Studio

1. SSH into the EMR master node:
   ```bash
   ssh -i your-key.pem hadoop@<master-node-dns>
   ```
2. Use Spark shell directly:
   ```bash
   spark-shell --master yarn
   ```

---

## Step 8: Environment Variables

Set these in your local environment or in EMR Studio:

```bash
# Windows PowerShell:
$env:S3_BUCKET="your-bucket-name"
$env:AWS_REGION="us-east-1"

# Or create .env file:
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
```

---

## Troubleshooting

### PassRole Error Still Appears

1. **Check IAM user permissions:**

   ```bash
   aws iam get-user
   aws iam list-attached-user-policies --user-name YOUR_USERNAME
   ```

2. **Verify role exists:**

   ```bash
   aws iam get-role --role-name EMR_DefaultRole
   ```

3. **Check trust relationships:**
   - Role must trust EMR service
   - Execution role must trust EMR Studio

### S3 Access Denied

1. **Check bucket policy:**

   ```bash
   aws s3api get-bucket-policy --bucket $BUCKET_NAME
   ```

2. **Verify IAM role has S3 permissions**

### Cluster Won't Start

1. **Check EC2 instance limits**
2. **Verify key pair exists**
3. **Check VPC/subnet configuration**
4. **Review CloudWatch logs**

---

## Cost Optimization

- **Use Spot Instances:** Add `--instance-market-type SPOT` when creating clusters
- **Auto-terminate:** Always enable auto-terminate
- **Use appropriate instance types:** m5.large is sufficient for development
- **Terminate when done:** Don't leave clusters running overnight

---

## Next Steps

Once AWS is set up:

1. **Test data profiling** (Hussain's code):

   ```python
   from src.hussain.profiling.data_profiler import CreditCardDataProfiler
   # ... run profiling on EMR
   ```

2. **Run feature engineering** (Rohit's code):

   ```python
   from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
   # ... run on EMR
   ```

3. **Train models and evaluate** (All team members' code)

---

## Support

If you encounter issues:

1. Check AWS CloudWatch logs
2. Review EMR cluster logs in S3
3. Check IAM permissions carefully
4. Verify all roles and policies are correctly set up

Good luck with your setup! 🚀
