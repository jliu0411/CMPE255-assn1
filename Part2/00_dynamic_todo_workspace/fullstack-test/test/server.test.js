import test from 'node:test'; import assert from 'node:assert/strict'; import { normalizeTask } from '../server.js';
test('normalizes a valid task',()=>{const task=normalizeTask({title:'  Ship it  ',priority:'high',tags:['Work','Work','']});assert.equal(task.title,'Ship it');assert.equal(task.priority,'high');assert.deepEqual(task.tags,['Work']);assert.equal(task.status,'todo')});
test('rejects an empty title',()=>assert.throws(()=>normalizeTask({title:' '}),/required/));
test('preserves identity and creation timestamp on update',()=>{const original=normalizeTask({title:'First'});const updated=normalizeTask({title:'Second',status:'done'},original);assert.equal(updated.id,original.id);assert.equal(updated.createdAt,original.createdAt);assert.ok(updated.completedAt)});
