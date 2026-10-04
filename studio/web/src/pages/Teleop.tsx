import { ActionBar } from "../components/ActionBar";
import { ArmsPanel } from "../components/ArmsPanel";
import { Cameras } from "../components/Cameras";
import { JointTable } from "../components/JointTable";
import { MockPanel } from "../components/MockPanel";
import { Notices } from "../components/Notices";

export function Teleop() {
  return (
    <div className="page teleop">
      <ActionBar activity="teleop" />
      <Notices />
      <div className="work-grid">
        <div className="col">
          <Cameras />
          <JointTable />
        </div>
        <div className="col rail">
          <ArmsPanel />
          <MockPanel />
        </div>
      </div>
    </div>
  );
}
